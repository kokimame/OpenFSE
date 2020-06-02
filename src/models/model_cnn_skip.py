import torch
import torch.nn as nn

# Convolution Neural Network with Skip connections as in the work:
# "A Single-Step Approach to Musical Tempo Estimation Using a Convolutional Neural Network"

def nn_conv2d_keep_size(
        in_channels, out_channels, kernel_size, padding=1, bias=True):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        bias=bias
    )

def nn_batch_normalization(
        num_features, bias=False):
    return nn.BatchNorm2d(
        num_features=num_features,
        bias=bias
    )

class PCNNModel(nn.Module):
    def __init__(self, emb_size=256):
        super().__init__()

        for i, (inp, outp) in enumerate([(1, 12), (13, 12), (25, 12), (12, 60), (72, 60),
                                       (132, 60), (60, 120), (120, 240), (240, emb_size)]):
            pass
