import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGModelV2(nn.Module):
    """
    Model object for MOVE.
    The explanation of the design choices can be found at https://arxiv.org/abs/1910.12551.
    """

    def __init__(self, emb_size=256):
        """
        Initializing the network
        :param emb_size: the size of the final embeddings produced by the model
        :param sum_method: the summarization method for the model
        :param final_activation: final activation to use for the model
        """
        super().__init__()

        self.prelu1 = nn.PReLU(init=0.01)
        self.prelu2 = nn.PReLU(init=0.01)
        self.prelu3 = nn.PReLU(init=0.01)
        self.prelu4 = nn.PReLU(init=0.01)
        self.prelu5 = nn.PReLU(init=0.01)
        self.prelu6 = nn.PReLU(init=0.01)

        self.num_of_channels = 16
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.num_of_channels,
                               kernel_size=(3, 3),
                               padding=1,
                               bias=True)
        nn.init.kaiming_normal_(self.conv1.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

        self.key_pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=self.num_of_channels,
                               out_channels=self.num_of_channels,
                               kernel_size=(3, 3),
                               padding=1,
                               bias=True)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

        self.conv3 = nn.Conv2d(in_channels=self.num_of_channels,
                               out_channels=self.num_of_channels,
                               kernel_size=(3, 3),
                               padding=1,
                               bias=True)
        nn.init.kaiming_normal_(self.conv3.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

        self.conv4 = nn.Conv2d(in_channels=self.num_of_channels,
                               out_channels=self.num_of_channels,
                               kernel_size=(3, 3),
                               padding=1,
                               bias=True)
        nn.init.kaiming_normal_(self.conv4.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

        self.conv5 = nn.Conv2d(in_channels=self.num_of_channels,
                               out_channels=self.num_of_channels,
                               kernel_size=(3, 3),
                               # dilation=(1, 13),
                               padding=1,
                               bias=True)
        nn.init.kaiming_normal_(self.conv5.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

        self.conv6 = nn.Conv2d(in_channels=self.num_of_channels,
                               out_channels=self.num_of_channels,
                               kernel_size=(3, 3),
                               # dilation=(1, 13),
                               padding=1,
                               bias=True)
        nn.init.kaiming_normal_(self.conv6.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

        self.fin_emb_size = emb_size

        self.autopool_p = nn.Parameter(torch.tensor(0.).float())


        self.lin_bn = nn.BatchNorm1d(emb_size, affine=False)

        self.lin1 = nn.Linear(in_features=4*self.num_of_channels, out_features=emb_size, bias=False)
        self.double()

    def forward(self, data):
        """
        Defining a forward pass of the network
        :param data: input tensor for the network
        :return: output tensor
        """
        x = self.prelu1(self.key_pool(self.conv1(data))) # 64x64x256
        x = self.prelu2(self.key_pool(self.conv2(x))) # 32x32x256
        x = self.prelu3(self.key_pool(self.conv3(x))) # 16x16x256
        x = self.prelu4(self.key_pool(self.conv4(x))) # 8x8x256
        x = self.prelu5(self.key_pool(self.conv5(x))) # 4x4x256
        x = self.prelu6(self.key_pool(self.conv6(x))) #2x2x256

        x = torch.flatten(x, start_dim=1)
        x = self.lin1(x)

        # Final activation
        # x = torch.sigmoid(x)
        # x = torch.tanh(x)
        x = self.lin_bn(x)

        return x

    def autopool_weights(self, data):
        """
        Calculating the autopool weights for a given tensor
        :param data: tensor for calculating the softmax weights with autopool
        :return: softmax weights with autopool
        """
        x = data * self.autopool_p
        max_values = torch.max(x, dim=3, keepdim=True).values
        softmax = torch.exp(x - max_values)
        weights = softmax / torch.sum(softmax, dim=3, keepdim=True)

        return weights
