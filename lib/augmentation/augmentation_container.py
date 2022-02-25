import torch
import torch.nn as nn


def slicd_Wasserstein_distance(x1, x2, n_projection=128):
    x1 = x1.flatten(-2).transpose(1, 2).contiguous() # (b, 3, h, w) -> (b, n, 3)
    x2 = x2.flatten(-2).transpose(1, 2).contiguous()
    rand_proj = torch.randn(3, n_projection, device=x1.device)
    rand_proj = rand_proj / (rand_proj.norm(2, dim=0, keepdim=True) + 1e-12)
    sorted_proj_x1 = torch.matmul(x1, rand_proj).sort(0)[0]
    sorted_proj_x2 = torch.matmul(x2, rand_proj).sort(0)[0]
    return (sorted_proj_x1 - sorted_proj_x2).pow(2).mean()


class AugmentationContainer(nn.Module):
    def __init__(
            self, c_aug, g_aug, c_reg_coef=0,
            normalizer=None, replay_buffer=None, n_chunk=16):
        super().__init__()
        self.c_aug = c_aug
        self.g_aug = g_aug
        self.c_reg_coef = c_reg_coef
        self.normalizer = normalizer
        self.replay_buffer = replay_buffer
        self.n_chunk = n_chunk

    def get_params(self, x, c, c_aug, g_aug):
        # sample noise vector from unit gauss
        noise = x.new(x.shape[0], self.g_aug.n_dim).normal_()
        target = self.normalizer(x) if self.normalizer is not None else x
        # sample augmentation parameters
        grid = g_aug(target, noise, c)
        scale, shift = c_aug(target, noise, c)
        return (scale, shift), grid

    def augmentation(self, x, c, c_aug, g_aug, update=False):
        c_param, g_param = self.get_params(x, c, c_aug, g_aug)
        # color augmentation
        aug_x = c_aug.transform(x, *c_param)
        # color regularization
        if update and self.c_reg_coef > 0:
            if self.normalizer is not None:
                swd = self.c_reg_coef * slicd_Wasserstein_distance(self.normalizer(x), self.normalizer(aug_x))
            else:
                swd = self.c_reg_coef * slicd_Wasserstein_distance(x, aug_x)
        else:
            swd = torch.zeros(1, device=x.device)
        # geometric augmentation
        aug_x = g_aug.transform(aug_x, g_param)
        return aug_x, swd

    def forward(self, x, c, update=False):
        if update or self.replay_buffer is None or len(self.replay_buffer) == 0:
            x, swd = self.augmentation(x, c, self.c_aug, self.g_aug, update)
        else:
            policies = self.replay_buffer.sampling(self.n_chunk, self.get_augmentation_model())
            if c is not None:
                x = torch.cat([self.augmentation(_x, _c, *policy, update)[0]
                                for _x, _c, policy in zip(x.chunk(self.n_chunk), c.chunk(self.n_chunk), policies)])
            else:
                x = torch.cat([self.augmentation(_x, None, *policy, update)[0]
                                for _x, policy in zip(x.chunk(self.n_chunk), policies)])

            swd = torch.zeros(1, device=x.device)
        return x, swd

    def get_augmentation_model(self):
        return nn.ModuleList([self.c_aug, self.g_aug])

    def reset(self):
        # initialize parameters
        self.c_aug.reset()
        self.g_aug.reset()
