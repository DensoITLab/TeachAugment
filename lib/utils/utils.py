import os
import sys
import yaml
import json
import logging

import torch


def set_seed(seed):
    import random
    import numpy.random
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def setup_ddp(args):
    torch.cuda.set_device(args.local_rank)
    if getattr(args, 'port', None) is not None:
        torch.distributed.init_process_group(
                backend='nccl',
                init_method=f'tcp://127.0.0.1:{args.port}',
                world_size=args.world_size,
                rank=args.local_rank,
            )
    else:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = int(os.environ['WORLD_SIZE'])
    args.num_workers //= args.world_size
    args.lr *= args.world_size
    return args


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def setup_logger(log_dir=None, resume=False):
    plain_formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s', datefmt='%m/%d %H:%M:%S'
    )
    logger = logging.getLogger() # root logger
    logger.setLevel(logging.INFO)
    s_handler = logging.StreamHandler(stream=sys.stdout)
    s_handler.setFormatter(plain_formatter)
    s_handler.setLevel(logging.INFO)
    logger.addHandler(s_handler)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        if not resume and os.path.exists(os.path.join(log_dir, 'console.log')):
            os.remove(os.path.join(log_dir, 'console.log'))
        f_handler = logging.FileHandler(os.path.join(log_dir, 'console.log'))
        f_handler.setFormatter(plain_formatter)
        f_handler.setLevel(logging.INFO)
        logger.addHandler(f_handler)


class AvgMeter:
    def __init__(self, ema_coef=0.9):
        self.ema_coef = ema_coef
        self.ema_params = {}
        self.sum_params = {}
        self.counter = {}

    def add(self, params:dict, ignores:list = []):
        for k, v in params.items():
            if k in ignores:
                continue
            if not k in self.ema_params.keys():
                self.ema_params[k] = v
                self.counter[k] = 1
            else:
                self.ema_params[k] -= (1 - self.ema_coef) * (self.ema_params[k] - v)
                self.counter[k] += 1
            if not k in self.sum_params.keys():
                self.sum_params[k] = v
            else:
                self.sum_params[k] += v

    def state(self, header="", footer="", ignore_keys=None):
        if ignore_keys is None:
            ignore_keys = set()
        state = header
        for k, v in self.ema_params.items():
            if k in ignore_keys:
                continue
            state += f" {k} {v:.6g} |"
        return state + " " + footer

    def mean_state(self, header="", footer=""):
        state = header
        for k, v in self.sum_params.items():
            state += f" {k} {v/self.counter[k]:.6g} |"
            self.counter[k] = 0
        state += footer

        self.sum_params = {}

        return state

    def reset(self):
        self.ema_params = {}
        self.sum_params = {}
        self.counter = {}


def override_config(args, dict_param):
    for k, v in dict_param.items():
        if isinstance(v, dict):
            args = override_config(args, v)
        else:
            setattr(args, k, v)
    return args


def load_yaml(path):
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    return d


def load_json(path):
    with open(path, 'r') as f:
        d = json.load(f)
    return d
