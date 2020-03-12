import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGModel(nn.Module):
    def __init__(self, emb_size=256, sum_method=4, final_activation=3):
        """
        :param emb_size: The dimension of output embedding
        :param sum_method: Not used right now
        :param final_activation: Not used right now
        """
        super().__init__()
        self.prelu1 = nn.PReLU(init=0.01)
        self.prelu2 = nn.PReLU(init=0.01)
        self.prelu3 = nn.PReLU(init=0.01)
        self.prelu4 = nn.PReLU(init=0.01)
        self.prelu5 = nn.PReLU(init=0.01)

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=16,
                               kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=16,
                               kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=16,
                               out_channels=16,
                               kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=16,
                               out_channels=16,
                               kernel_size=(3, 3))
        self.conv5 = nn.Conv2d(in_channels=16,
                               out_channels=16,
                               kernel_size=(3, 3))
        self.fin_emb_size = emb_size
        self.double()

    def forward(self, data):
        x = self.prelu1(self.maxpool(self.conv1(data)))
        x = self.prelu2(self.maxpool(self.conv2(x)))
        x = self.prelu3(self.maxpool(self.conv3(x)))
        x = self.prelu4(self.maxpool(self.conv4(x)))
        x = self.prelu5(self.maxpool(self.conv5(x)))

        x = torch.flatten(x)
        x = x.unsqueeze(0)
        return x