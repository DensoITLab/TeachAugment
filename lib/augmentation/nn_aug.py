import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def relaxed_bernoulli(logits, temp=0.05, device='cpu'):
    u = torch.rand_like(logits, device=device)
    l = torch.log(u) - torch.log(1 - u)
    return ((l + logits)/temp).sigmoid()


class TriangleWave(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        o = torch.acos(torch.cos(x * math.pi)) / math.pi
        self.save_for_backward(x)
        return o

    @staticmethod
    def backward(self, grad):
        o = self.saved_tensors[0]
        # avoid nan gradient at the peak by replacing it with the right derivative
        o = torch.floor(o) % 2
        grad[o == 1] *= -1 
        return grad


class ColorAugmentation(nn.Module):
    def __init__(self, n_classes=10, scale=1, hidden=128, n_dim=128, dropout_ratio=0.8, with_context=True):
        super().__init__()

        n_hidden = 4 * n_dim
        conv = lambda ic, io, k : nn.Conv2d(ic, io, k, padding=k//2, bias=False)
        linear = lambda ic, io : nn.Linear(ic, io, False)
        bn2d = lambda c : nn.BatchNorm2d(c, track_running_stats=False)
        bn1d = lambda c : nn.BatchNorm1d(c, track_running_stats=False)

        # embedding layer for context vector
        if with_context:
            self.context_layer = conv(n_classes, hidden, 1)
        else:
            self.context_layer = None
        # embedding layer for RGB
        self.color_enc1 = conv(3, hidden, 1)
        # body for RGB
        self.color_enc_body = nn.Sequential(
            bn2d(hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Sequential(),
            conv(hidden, hidden, 1),
            bn2d(hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Sequential()
        )
        # output layer for RGB
        self.c_regress = conv(hidden, 6, 1)
        # body for noise vector
        self.noise_enc = nn.Sequential(
            linear(n_dim + n_classes if with_context else n_dim, n_hidden),
            bn1d(n_hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Sequential(),
            linear(n_hidden, n_hidden),
            bn1d(n_hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Sequential(),
        )
        # output layer for noise vector
        self.n_regress = linear(n_hidden, 2)

        if with_context:
            self.register_parameter('logits', nn.Parameter(torch.zeros(n_classes)))
        else:
            self.register_parameter('logits', nn.Parameter(torch.zeros(1)))
        # initialize parameters
        self.reset()

        self.with_context = with_context
        self.n_classes = n_classes
        self.n_dim = n_dim
        self.scale = scale
        self.relax = True
        self.stochastic = True

    def sampling(self, scale, shift, y, temp=0.05):
        if self.stochastic: # random apply
            if self.with_context:
                logits = self.logits[y].reshape(-1, 1, 1, 1)
            else:
                logits = self.logits.repeat(scale.shape[0]).reshape(-1, 1, 1, 1)
            prob = relaxed_bernoulli(logits, temp, device=scale.device)
            if not self.relax: # hard sampling
                prob = (prob > 0.5).float()
            scale = 1 - prob + prob * scale
            shift = prob * shift # omit "+ (1 - prob) * 0"
        return scale, shift

    def forward(self, x, noise, c=None):
        if self.with_context:
            # integer to onehot vector
            onehot_c = nn.functional.one_hot(c, self.n_classes).float()
            noise = torch.cat([onehot_c, noise], 1)
            # vector to 2d image
            onehot_c = onehot_c.reshape(*onehot_c.shape, 1, 1)
        # global scale and shift
        gfactor = self.noise_enc(noise)
        gfactor = self.n_regress(gfactor).reshape(-1, 2, 1, 1)
        # per-pixel scale and shift
        feature = self.color_enc1(x)
        # add context information
        if self.with_context:
            feature = self.context_layer(onehot_c) + feature
        feature = self.color_enc_body(feature)
        factor = self.c_regress(feature)
        # add up parameters
        scale, shift = factor.chunk(2, dim=1)
        g_scale, g_shift = gfactor.chunk(2, dim=1)
        scale = (g_scale + scale).sigmoid()
        shift = (g_shift + shift).sigmoid()
        # scaling
        scale = self.scale * (scale - 0.5) + 1
        shift = shift - 0.5
        # random apply
        scale, shift = self.sampling(scale, shift, c)

        return scale, shift

    def reset(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, 0.2, 'fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # zero initialization
        nn.init.constant_(self.c_regress.weight, 0)
        nn.init.constant_(self.n_regress.weight, 0)
        nn.init.constant_(self.logits, 0)

    def transform(self, x, scale, shift):
        # ignore zero padding region
        with torch.no_grad():
            h, w = x.shape[-2:]
            mask = (x.sum(1, keepdim=True) == 0).float() # mask pixels having (0, 0, 0) color
            mask = torch.logical_and(mask.sum(-1, keepdim=True) < w,
                                     mask.sum(-2, keepdim=True) < h) # mask zero padding region

        x = (scale * x + shift) * mask
        return TriangleWave.apply(x)
        

class GeometricAugmentation(nn.Module):
    def __init__(self, n_classes=10, scale=0.5, n_dim=128, dropout_ratio=0.8, with_context=True):
        super().__init__()

        hidden = 4 * n_dim
        linear = lambda ic, io : nn.Linear(ic, io, False)
        bn1d = lambda c : nn.BatchNorm1d(c, track_running_stats=False)

        self.body = nn.Sequential(
            linear(n_dim + n_classes if with_context else n_dim, hidden),
            bn1d(hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Sequential(),
            linear(hidden, hidden),
            bn1d(hidden),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Sequential(),
        )

        self.regressor = linear(hidden, 6)
        # identity matrix
        self.register_buffer('i_matrix', torch.Tensor([[1, 0, 0], [0, 1, 0]]).reshape(1, 2, 3))

        if with_context:
            self.register_parameter('logits', nn.Parameter(torch.zeros(n_classes)))
        else:
            self.register_parameter('logits', nn.Parameter(torch.zeros(1)))
        # initialize parameters
        self.reset()

        self.with_context = with_context
        self.n_classes = n_classes
        self.n_dim = n_dim
        self.scale = scale

        self.relax = True
        self.stochastic = True

    def sampling(self, A, y=None, temp=0.05):
        if self.stochastic: # random apply
            if self.with_context:
                logits = self.logits[y].reshape(-1, 1, 1)
            else:
                logits = self.logits.repeat(A.shape[0]).reshape(-1, 1, 1)
            prob = relaxed_bernoulli(logits, temp, device=logits.device)
            if not self.relax: # hard sampling
                prob = (prob > 0.5).float()
            return ((1 - prob) * self.i_matrix + prob * A)
        else:
            return A

    def forward(self, x, noise, c=None):
        if self.with_context:
            with torch.no_grad():
                # integer to onehot vector
                onehot_c = nn.functional.one_hot(c, self.n_classes).float()
                noise = torch.cat([onehot_c, noise], 1)
        features = self.body(noise)
        A = self.regressor(features).reshape(-1, 2, 3)
        # scaling
        A = self.scale * (A.sigmoid() - 0.5) + self.i_matrix
        # random apply
        A = self.sampling(A, c)
        # matrix to grid representation
        grid = nn.functional.affine_grid(A, x.shape)
        return grid

    def reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, 0.2, 'fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # zero initialization
        nn.init.constant_(self.regressor.weight, 0)
        nn.init.constant_(self.logits, 0)

    def transform(self, x, grid):
        x = F.grid_sample(x, grid, mode='bilinear')
        return x
