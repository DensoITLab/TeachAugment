import random

import torch


def _gen_cutout_coord(height, width, size):
    height_loc = random.randint(0, height - 1)
    width_loc = random.randint(0, width - 1)

    upper_coord = (max(0, height_loc - size // 2),
                    max(0, width_loc - size // 2))
    lower_coord = (min(height, height_loc + size // 2),
                    min(width, width_loc + size // 2))

    return upper_coord, lower_coord


class Cutout(torch.nn.Module):
    def __init__(self, size=16):
        super().__init__()
        self.size = size

    def forward(self, img):
        h, w = img.shape[-2:]
        upper_coord, lower_coord  = _gen_cutout_coord(h, w, self.size)

        mask_height = lower_coord[0] - upper_coord[0]
        mask_width = lower_coord[1] - upper_coord[1]
        assert mask_height > 0
        assert mask_width > 0

        mask = torch.ones_like(img)
        mask[..., upper_coord[0]:lower_coord[0], upper_coord[1]:lower_coord[1]] = 0
        return img * mask
