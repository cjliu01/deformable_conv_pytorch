from deform_conv import DeformConv2D
import torch
from torch import nn

if __name__ == '__main__':
    a = torch.randn((1, 3, 224, 224))
    conv = DeformConv2D(3, 128, kernel_size=3, padding=1, modulation=True)
    print(conv(a).shape)