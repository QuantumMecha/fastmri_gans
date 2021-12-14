from collections import OrderedDict

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32, ngpu=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        features = init_features
        self.encoder1 = Generator._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = Generator._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = Generator._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = Generator._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = Generator._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = Generator._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = Generator._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = Generator._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = Generator._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        #print('input', x.shape)
        enc1 = self.encoder1(x)
        #print('enc1', enc1.shape)
        enc2 = self.encoder2(self.pool1(enc1))
        #print('enc2', enc2.shape)
        enc3 = self.encoder3(self.pool2(enc2))
        #print('enc3', enc3.shape)
        enc4 = self.encoder4(self.pool3(enc3))
        #print('enc4', enc4.shape)
        bottleneck = self.bottleneck(self.pool4(enc4))
        #print('bottleneck', bottleneck.shape)
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        #print('dec4', dec4.shape)
        dec3 = self.upconv3(dec4)
        #print('dec3', dec3.shape)
        dec3 = torch.cat((dec3, enc3), dim=1)
        #print('dec3', dec3.shape)
        dec3 = self.decoder3(dec3)
        #print('dec3', dec3.shape)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        #print('dec2', dec2.shape)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        #print('dec1', dec1.shape)
        #print('sig', torch.sigmoid(self.conv(dec1)).shape)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=32, ngpu=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.network = nn.Sequential(
            # input is (in_channels) x 64 x 64
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features) x 32 x 32
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features*2) x 16 x 16
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features*4) x 8 x 8
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features*8) x 4 x 4
            nn.Flatten(),
            nn.Dropout(0.3),
            #nn.Linear(features * 8, 1),
            nn.Linear(82944, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        #print(input.shape) # Debug
        out = self.network(input)
        return out
