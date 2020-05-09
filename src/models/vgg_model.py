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
        self.prelu6 = nn.PReLU(init=0.01)

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=16,
                               kernel_size=(3, 3),
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=16,
                               kernel_size=(3, 3),
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=16,
                               out_channels=16,
                               kernel_size=(3, 3),
                               padding=1)
        self.conv4 = nn.Conv2d(in_channels=16,
                               out_channels=16,
                               kernel_size=(3, 3),
                               padding=1)
        self.conv5 = nn.Conv2d(in_channels=16,
                               out_channels=16,
                               kernel_size=(3,3),
                               padding=1)
        # 5 conv without padding is possible too!
        self.conv6 = nn.Conv2d(in_channels=16,
                               out_channels=16,
                               kernel_size=(3,3),
                               padding=1)
        self.lin1 = nn.Linear(in_features=64, out_features=emb_size, bias=True)
        self.lin_bn = nn.BatchNorm1d(emb_size, affine=False)

        self.fin_emb_size = emb_size
        self.double()

    def forward(self, data):
        x = self.prelu1(self.maxpool(self.conv1(data))) # 64x64x16
        x = self.prelu2(self.maxpool(self.conv2(x))) # 32x32x16
        x = self.prelu3(self.maxpool(self.conv3(x))) # 16x16x16
        x = self.prelu4(self.maxpool(self.conv4(x))) # 8x8x16
        x = self.prelu5(self.maxpool(self.conv5(x))) # 4x4x16
        x = self.prelu6(self.maxpool(self.conv6(x))) # 2x2x16

        x = torch.flatten(x, start_dim=1)
        x = self.lin1(x)
        x = self.lin_bn(x)
        return x