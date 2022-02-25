import os
import logging
import warnings

logger = logging.getLogger(__name__)
warnings.simplefilter('ignore', UserWarning)

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.utils import save_image

from lib import augmentation, build_dataset, teachaugment
from lib.utils import utils, lr_scheduler
from lib.models import build_model
from lib.losses import non_saturating_loss


def main(args):
    main_process = args.local_rank == 0
    if main_process:
        logger.info(args)
    # Setup GPU
    if torch.cuda.is_available():
        device = 'cuda'
        if args.disable_cudnn:
            # torch.nn.functional.grid_sample, which is used for geometric augmentation, is non-deterministic
            # so, reproducibility is not ensured even though following option is True
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.benchmark = True
    else:
        raise NotImplementedError('CUDA is unavailable.')
    # Dataset
    base_aug, train_trans, val_trans, normalizer = augmentation.get_transforms(args.dataset)
    train_data, eval_data, n_classes = build_dataset(args.dataset, args.root, train_trans, val_trans)
    sampler = torch.utils.data.DistributedSampler(train_data, num_replicas=args.world_size, rank=args.local_rank) if args.dist else None
    train_loader = DataLoader(train_data, args.batch_size, not args.dist, sampler,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    eval_loader = DataLoader(eval_data, 1)
    # Model
    model = build_model(args.model, n_classes).to(device)
    model.train()
    # EMA Teacher
    avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
                args.ema_rate * averaged_model_parameter + (1 - args.ema_rate) * model_parameter
    ema_model = optim.swa_utils.AveragedModel(model, avg_fn=avg_fn)
    for ema_p in ema_model.parameters():
        ema_p.requires_grad_(False)
    ema_model.train()
    # Trainable Augmentation
    rbuffer = augmentation.replay_buffer.ReplayBuffer(args.rb_decay)
    trainable_aug = augmentation.build_augmentation(n_classes, args.g_scale, args.c_scale,
                                                    args.c_reg_coef, normalizer, rbuffer,
                                                    args.batch_size // args.group_size,
                                                    not args.wo_context).to(device)
    # Baseline augmentation
    base_aug = torch.nn.Sequential(*base_aug).to(device)
    if main_process:
        logger.info('augmentation')
        logger.info(trainable_aug)
        logger.info(base_aug)
    # Optimizer
    optim_cls = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0)
    optim_aug = optim.AdamW(trainable_aug.parameters(), lr=args.aug_lr, weight_decay=args.aug_weight_decay)
    if args.dataset == 'ImageNet':
        scheduler = lr_scheduler.MultiStepLRWithLinearWarmup(optim_cls, 5, [90, 180, 240], 0.1)
    else:
        scheduler = lr_scheduler.CosineAnnealingWithLinearWarmup(optim_cls, 5, args.n_epochs)

    # Following Fast AutoAugment (https://github.com/kakaobrain/fast-autoaugment),
    # pytorch-gradual-warmup-lr (https://github.com/ildoonet/pytorch-gradual-warmup-lr) was used for the paper experiments.
    # The implementation of our "*WithLinearWarmup" is slightly different from GradualWarmupScheduler.
    # Thus, to reproduce experimental results strictly, please use following scheduler, instead of above scheduler.

    # Don't forget to install pytorch-gradual-warmup-lr
    #     pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git

    # from warmup_scheduler import GradualWarmupScheduler
    # if args.dataset == 'ImageNet':
    #     base_scheduler = optim.lr_scheduler.MultiStepLR(optim_cls, [90, 180, 240], 0.1)
    #     scheduler = GradualWarmupScheduler(optim_cls, 1, 5, base_scheduler)
    # else:
    #     base_scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_cls, args.n_epochs)
    #     scheduler = GradualWarmupScheduler(optim_cls, 1, 5, base_scheduler)

    # Objective function
    adv_criterion = non_saturating_loss.NonSaturatingLoss(args.epsilon)
    objective = teachaugment.TeachAugment(model, ema_model, trainable_aug,
                                          adv_criterion, args.weight_decay,
                                          base_aug, normalizer, not args.dist and args.save_memory).to(device)
    # DDP
    if args.dist:
        objective = torch.nn.parallel.DistributedDataParallel(objective, device_ids=[args.local_rank],
                                                              output_device=args.local_rank, find_unused_parameters=True)
    # Resume
    st_epoch = 1
    if args.resume:
        checkpoint = torch.load(os.path.join(args.log_dir, 'checkpoint.pth'))
        st_epoch += checkpoint['epoch']
        if main_process:
            logger.info(f'resume from epoch {st_epoch}')
        buffer_length = checkpoint['epoch'] // args.sampling_freq
        rbuffer.initialize(buffer_length, trainable_aug.get_augmentation_model()) # define placeholder for load_state_dict
        objective.load_state_dict(checkpoint['objective']) # including model, ema teacher, trainable_aug, and replay buffer
        optim_cls.load_state_dict(checkpoint['optim_cls'])
        optim_aug.load_state_dict(checkpoint['optim_aug'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    # Training loop
    if main_process:
        logger.info('training')
    meter = utils.AvgMeter()
    for epoch in range(st_epoch, args.n_epochs + 1):
        if args.dist:
            train_loader.sampler.set_epoch(epoch)
        for i, data in enumerate(train_loader):
            torch.cuda.synchronize()
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            if args.wo_context:
                context = None
            else:
                context = targets
            # update teacher model
            ema_model.update_parameters(model)
            # Update augmentation
            if i % args.n_inner == 0:
                optim_aug.zero_grad()
                if args.dist and args.save_memory: # computating gradient independently for saving memory
                    loss_adv, c_reg, acc_tar = objective(inputs, targets, context, 'loss_adv')
                    (loss_adv + 0.5 * c_reg).backward()
                    loss_tea, c_reg, acc_tea = objective(inputs, targets, context, 'loss_tea')
                    (loss_tea + 0.5 * c_reg).backward()
                    res = {'loss adv.': loss_adv.item(),
                           'loss teacher': loss_tea.item(),
                           'color reg.': c_reg.item(),
                           'acc.': acc_tar.item(),
                           'acc. teacher': acc_tea.item()}
                else:
                    loss_aug, res = objective(inputs, targets, context, 'aug')
                    loss_aug.backward()
                optim_aug.step()
                meter.add(res)
            # Update target model
            optim_cls.zero_grad()
            loss_cls, res, aug_img = objective(inputs, targets, context, 'cls')
            loss_cls.backward()
            if args.dataset != 'ImageNet':
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim_cls.step()
            # Adjust learning rate
            scheduler.step(epoch - 1. + (i + 1.) / len(train_loader))
            # Print losses and accuracy
            meter.add(res)
            if main_process and (i + 1) % args.print_freq == 0:
                logger.info(meter.state(f'epoch {epoch} [{i+1}/{len(train_loader)}]',
                                        f'lr {optim_cls.param_groups[0]["lr"]:.4e}'))
        # Store augmentation in buffer
        if args.sampling_freq > 0 and epoch % args.sampling_freq == 0:
            rbuffer.store(trainable_aug.get_augmentation_model())
            if main_process:
                logger.info(f'store augmentation (buffer length: {len(rbuffer)})')
        # Save checkpoint
        if main_process:
            logger.info(meter.mean_state(f'epoch [{epoch}/{args.n_epochs}]',
                                         f'lr {optim_cls.param_groups[0]["lr"]:.4e}'))
            checkpoint = {'model': model.state_dict(),
                          'objective': objective.state_dict(), # including ema model and replay buffer
                          'optim_cls': optim_cls.state_dict(),
                          'optim_aug': optim_aug.state_dict(),
                          'scheduler': scheduler.state_dict(),
                          'epoch': epoch}
            torch.save(checkpoint, os.path.join(args.log_dir, 'checkpoint.pth'))
        # Save augmented images
        if args.vis:
            save_image(aug_img, os.path.join(args.log_dir, f'{epoch}epoch_aug_img.png'))
            save_image(inputs, os.path.join(args.log_dir, f'{epoch}epoch_img.png'))
    # Evaluation
    if main_process:
        logger.info('evaluation')
        acc1, acc5 = 0, 0
        model.eval()
        n_samples = len(eval_loader)
        with torch.no_grad():
            for data in eval_loader:
                input, target = data
                output = model(input.to(device))
                accs = utils.accuracy(output, target.to(device), (1, 5))
                acc1 += accs[0]
                acc5 += accs[1]
        logger.info(f'{args.dataset} error rate (%) | Top1 {100 - acc1/n_samples} | Top5 {100 - acc5/n_samples}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'ImageNet'])
    parser.add_argument('--root', default='./data', type=str,
                        help='/path/to/dataset')
    # Model
    parser.add_argument('--model', default='wrn-28-10', type=str)
    # Optimization
    parser.add_argument('--lr', default=0.1, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', '-wd', default=5e-4, type=float)
    parser.add_argument('--n_epochs', default=200, type=int)
    parser.add_argument('--batch_size', '-bs', default=128, type=int)
    parser.add_argument('--aug_lr', default=1e-3, type=float,
                        help='learning rate for augmentation model')
    parser.add_argument('--aug_weight_decay', '-awd', default=1e-2, type=float,
                        help='weight decay for augmentation model')
    # Augmentation
    parser.add_argument('--g_scale', default=0.5, type=float,
                        help='the search range of the magnitude of geometric augmantation')
    parser.add_argument('--c_scale', default=0.8, type=float,
                        help='the search range of the magnitude of color augmantation')
    parser.add_argument('--group_size', default=8, type=int)
    parser.add_argument('--wo_context', action='store_true',
                        help='without context vector as input')
    # TeachAugment
    parser.add_argument('--n_inner', default=5, type=int,
                        help='the number of iterations for inner loop (i.e., updating classifier)')
    parser.add_argument('--ema_rate', default=0.999, type=float,
                        help='decay rate for the ema teacher')
    # Improvement techniques
    parser.add_argument('--c_reg_coef', default=10, type=float,
                        help='coefficient of the color regularization')
    parser.add_argument('--rb_decay', default=0.9, type=float,
                        help='decay rate for replay buffer')
    parser.add_argument('--sampling_freq', default=10, type=int,
                        help='sampling augmentation frequency')
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help='epsilon for the label smoothing')
    # Distributed data parallel
    parser.add_argument('--dist', action='store_true',
                        help='use distributed data parallel')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--world_size', '-ws', default=1, type=int)
    parser.add_argument('--port', default=None, type=str)
    # Misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--log_dir', default='./log', type=str)
    parser.add_argument('--disable_cudnn', action='store_true',
                        help='disable cudnn for reproducibility')
    parser.add_argument('--resume', action='store_true',
                        help='resume training')
    parser.add_argument('--num_workers', '-j', default=8, type=int,
                        help='the number of data loading workers')
    parser.add_argument('--vis', action='store_true',
                        help='visualize augmented images')
    parser.add_argument('--save_memory', action='store_true',
                        help='independently calculate adversarial loss \
                            and teacher loss for saving memory')
    parser.add_argument('--yaml', default=None, type=str,
                        help='given path to .json, parse from .yaml')
    parser.add_argument('--json', default=None, type=str,
                        help='given path to .json, parse from .json')

    args = parser.parse_args()

    # override args
    if args.yaml is not None:
        yaml_cfg = utils.load_yaml(args.yaml)
        args = utils.override_config(args, yaml_cfg)
    if args.json is not None:
        json_cfg = utils.load_json(args.json)
        args = utils.override_config(args, json_cfg)

    utils.set_seed(args.seed)
    if args.local_rank == 0:
        utils.setup_logger(args.log_dir, args.resume)
    if args.dist:
        utils.setup_ddp(args)

    main(args)
