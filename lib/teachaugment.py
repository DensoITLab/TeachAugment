import torch
import torch.nn as nn
import torch.nn.functional as F


class TeachAugment(nn.Module):
    """
    Args:
        model: nn.Module
            the target model
        ema_model: nn.Module
            exponential moving average of the target model
        trainable_aug: nn.Module
            augmentation model
        adv_criterion:
            criterion for the adversarial loss
        weight_decay: float
            coefficient for weight decay
        base_aug: nn.Module
            baseline augmentation
    """
    def __init__(
            self, model, ema_model,
            trainable_aug, adv_criterion,
            weight_decay=0, base_aug=None,
            normalizer=None, save_memory=False):
        super().__init__()
        # model
        self.model = model
        self.ema_model = ema_model
        # loss
        self.adv_criterion = adv_criterion
        self.weight_decay = weight_decay
        # augmentation
        self.trainable_aug = trainable_aug
        self.base_aug = base_aug
        self.normalizer = normalizer
        # misc
        self.save_memory = save_memory

    def manual_weight_decay(self, model, coef):
        return coef * (1. / 2.) * sum([params.pow(2).sum()
                                       for name, params in model.named_parameters()
                                       if not ('_bn' in name or '.bn' in name)])

    def forward(self, x, y, c=None, loss='cls'):
        """
        Args:
            x: torch.Tensor
                images
            y: torch.Tensor
                labels
            c: torch.Tensor
                context vector (optional)
            loss: str
                loss type
                - cls
                    loss for the target model
                - aug
                    loss for the augmentation model (sum of following loss_adv and loss_tea)
                - loss_adv
                    adversarial loss
                - loss_tea
                    loss for the teacher model
        """
        if loss == 'cls':
            return self.loss_classifier(x, y, c)
        elif loss == 'aug':
            return self.loss_augmentation(x, y, c)
        elif loss == 'loss_adv':
            return self.loss_adversarial(x, y, c)
        elif loss == 'loss_tea':
            return self.loss_teacher(x, y, c)
        else:
            raise NotImplementedError

    def loss_classifier(self, x, y, c=None):
        # augmentation
        with torch.no_grad():
            aug_x, _ = self.trainable_aug(x, c)
            if self.base_aug is not None:
                inputs = torch.stack([self.base_aug(_x) for _x in aug_x])
            else:
                inputs = aug_x
        # calculate loss
        pred = self.model(inputs)
        loss = F.cross_entropy(pred, y)
        res = {'loss cls.': loss.item()}
        if self.weight_decay > 0:
            loss += self.manual_weight_decay(self.model, self.weight_decay)
        return loss, res, aug_x

    def loss_augmentation(self, x, y, c=None):
        # avoid updating bn running stats because the stats has been updated in loss_classifier.
        self.stop_bn_track_running_stats(self.model)
        # augmentation
        x, c_reg = self.trainable_aug(x, c, update=True)
        if self.normalizer is not None:
            x = self.normalizer(x)
        # calculate loss
        tar_pred = self.model(x)
        loss_adv = self.adv_criterion(tar_pred, y)
        # compute gradient to release the memory for the computational graph
        # NOTE: save_memory does NOT work for DDP.
        # Under DDP, computing loss_tea and loss_adv independently using loss_teacher and loss_adversarial
        # see main.py l130-l138 for more details
        if self.save_memory:
            grad = torch.autograd.grad(loss_adv, x)[0]
            x.backward(grad, retain_graph=True)
        tea_pred = self.ema_model(x)
        loss_tea = F.cross_entropy(tea_pred, y)
        # accuracy
        with torch.no_grad():
            teacher_acc = (tea_pred.argmax(1) == y).float().mean()
            target_acc = (tar_pred.argmax(1) == y).float().mean()

        res = {'loss adv.': loss_adv.item(),
               'loss teacher': loss_tea.item(),
               'color reg.': c_reg.item(),
               'acc.': target_acc.item(),
               'acc. teacher': teacher_acc.item()}

        self.activate_bn_track_running_stats(self.model)

        if self.save_memory:
            return loss_tea + c_reg, res

        return loss_adv + loss_tea + c_reg, res

    def loss_adversarial(self, x, y, c=None):
        # avoid updating bn running stats twice with the same samples
        self.stop_bn_track_running_stats(self.model)
        # augmentation
        x, c_reg = self.trainable_aug(x, c, update=True)
        if self.normalizer is not None:
            x = self.normalizer(x)
        # calculate loss
        tar_pred = self.model(x)
        loss_adv = self.adv_criterion(tar_pred, y)
        # accuracy
        with torch.no_grad():
            acc = (tar_pred.argmax(1) == y).float().mean()

        self.activate_bn_track_running_stats(self.model)

        return loss_adv, c_reg, acc

    def loss_teacher(self, x, y, c=None):
        # augmentation
        x, c_reg = self.trainable_aug(x, c, update=True)
        if self.normalizer is not None:
            x = self.normalizer(x)
        # calculate loss
        tea_pred = self.ema_model(x)
        loss_tea = F.cross_entropy(tea_pred, y)
        # accuracy
        with torch.no_grad():
            acc = (tea_pred.argmax(1) == y).float().mean()
        
        return loss_tea, c_reg, acc

    def stop_bn_track_running_stats(self, model):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.track_running_stats = False

    def activate_bn_track_running_stats(self, model):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.track_running_stats = True
