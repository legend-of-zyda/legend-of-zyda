"""Saves the head and encoder of the imitation model to separate files.

Usage:
    # cd to the `helpers` directory (the one where this file is) and run:
    python freeze_head.py
"""
import os

import torch
import torch.nn.functional as F
from torch import nn

###############################################################
# # DEFINITION OF IMITATION LEARNING MODEL FOR REFERENCE
# class BasicConv2d(nn.Module):
#     def __init__(self, input_dim, output_dim, kernel_size, bn):
#         super().__init__()
#         self.conv = nn.Conv2d(
#             input_dim, output_dim,
#             kernel_size=kernel_size,
#             padding=(kernel_size[0] // 2, kernel_size[1] // 2)
#         )
#         self.bn = nn.BatchNorm2d(output_dim) if bn else None

#     def forward(self, x):
#         h = self.conv(x)
#         h = self.bn(h) if self.bn is not None else h
#         return h

# class LuxNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         layers, filters = 12, 32
#         self.conv0 = BasicConv2d(20, filters, (3, 3), True)
#         self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
#         self.head_p = nn.Linear(filters, 5, bias=False)

#     def forward(self, x):
#         h = F.relu_(self.conv0(x))
#         for block in self.blocks:
#             h = F.relu_(h + block(h))
#         h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
#         p = self.head_p(x)
#         return p

############################################################
## Extract modules from model and form encoder / controller

HERE = os.path.dirname(os.path.abspath(__file__))
FILENAME = os.path.join(HERE, '../agents/model_imitation_v2.pth')
MODEL = torch.jit.load(FILENAME)


class LuxEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        for name, module in MODEL.named_modules():
            if name == 'conv0':
                self.conv0 = module
            elif name == 'blocks':
                self.blocks = module

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks.children():
            h = F.relu_(h + block(h))
        h_head = (h * x[:, :1]).view(h.size(0), h.size(1), -1).sum(-1)
        return h_head


def main():
    for name, module in MODEL.named_modules():
        if name == 'head_p':
            head_p = module

    head_p.save('controller_model.pth')
    torch.save(head_p.state_dict(), 'controller_weights.pth')

    encoder = torch.jit.script(LuxEncoder())
    encoder.eval()
    torch.jit.save(torch.jit.freeze(encoder), 'state_encoder.pth')


if __name__ == "__main__":
    main()
