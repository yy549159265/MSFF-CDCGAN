import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5):
        super(Inception, self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1,stride=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=5,stride=1, padding=2),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True),

        )


    def forward(self, x):
        return torch.cat([self.b3(x), self.b2(x), self.b1(x)], dim=1)


class ResidualBlockc(nn.Module):
    def __init__(self, channels):
        super(ResidualBlockc, self).__init__()
        self.conv1 = nn.ConvTranspose2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)


        return  residual

class ResidualBlockd(nn.Module):
    def __init__(self, channels):
        super(ResidualBlockd, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)


        return  residual

class G_cdcgan(nn.Module):
    def __init__(self, d=64):
        super(G_cdcgan, self).__init__()

        self.feature = Inception(1, 1, 1, 1,1, 1)

        self.c1 = nn.Sequential(
            nn.Conv2d(4, d, 4, 2, 1),  # 8
            nn.BatchNorm2d(d),
            nn.PReLU(),
        )
        self.c1_r = ResidualBlockd(d)

        self.c2 = nn.Sequential(
            nn.Conv2d(d, d * 2, 4, 2, 1),  # 4
            nn.BatchNorm2d(d * 2),
            nn.PReLU(),
        )
        self.c2_r = ResidualBlockd(d*2)
        self.c3 = nn.Sequential(
            nn.Conv2d(d * 2, d*4,  4, 2, 1),
            nn.BatchNorm2d(d * 4),
            nn.PReLU(),
        )
        self.c3_r = ResidualBlockd(d * 4)

        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(d*4, d * 2, 4, 2, 1),
            nn.BatchNorm2d(d * 2),
            nn.PReLU(),
        )
        self.d1_r = ResidualBlockd(d * 2)

        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(d * 2, d, 4, 2, 1),
            nn.BatchNorm2d(d),
            nn.PReLU(),
        )
        self.d2_r = ResidualBlockd(d)

        self.d3 = nn.Sequential(
            nn.ConvTranspose2d(d, 1, 4, 2, 1),
            nn.Tanh()
        )


    def forward(self, c, x):

        # x = x.view(x.size(0), 8 * 8, x.size(2) // 8, x.size(3) // 8)
        # c = c.view(c.size(0), 8 * 8, c.size(2) // 8, c.size(3) // 8)



        c = self.feature(c)
        y = torch.cat([x, c], 1)
        # y = self.feature(y)
        # print(y.shape)
        # y = y.view(y.size(0),4*8 * 8, y.size(2) // 8, y.size(3) // 8)
        y = self.c1(y)
        y = self.c1_r(y)
        y = self.c2(y)
        y = self.c2_r(y)
        y = self.c3(y)

        y = self.d1(y)
        y = self.d1_r(y)
        y = self.d2(y)
        y = self.d2_r(y)
        y = self.d3(y)
        # print(y.shape)
        return y

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # 判别器（全卷积）


class D_cdcgan(nn.Module):
    def __init__(self, d=64):
        super(D_cdcgan, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(2, d, 4, 2, 1),  # 8
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2,inplace=True),
        )
        self.c1_r = ResidualBlockd(d)

        self.c2 = nn.Sequential(
            nn.Conv2d(d, d * 2, 4, 2, 1),  # 4
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(0.2,inplace=True),
        )
        self.c2_r = ResidualBlockd(d * 2)
        self.c3 = nn.Sequential(
            nn.Conv2d(d * 2, 1, 4, 2, 1),
            nn.Sigmoid()
        )




    def forward(self, c, x):
        y = torch.cat([x, c], dim=1)
        y = self.c1(y)
        y = self.c1_r(y)
        y = self.c2(y)
        y = self.c2_r(y)
        y = self.c3(y)
        # print(y.shape)
        return y

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
