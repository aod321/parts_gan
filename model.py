import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 12, kernel_size=3, stride=2, padding=1),  # 12 x 256 x 256
            nn.BatchNorm2d(12),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(12, 12, kernel_size=3, stride=2, padding=1),  # 12 x 128 x 128
            nn.BatchNorm2d(12),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(12, 12, kernel_size=3, stride=2, padding=1),  # 12 x 64 x 64
            nn.BatchNorm2d(12),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(12, 6, kernel_size=3, stride=1, padding=1),  # 6 x 64 x 64
            nn.BatchNorm2d(6),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.parts_block = nn.ModuleList([nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1),    # 3 x 64 x 64
            nn.Tanh()
        )
            for _ in range(4)])

    def forward(self, x):
        img = self.model(x)
        parts = torch.stack([self.parts_block[i](img)
                             for i in range(4)], dim=0)
        parts = torch.transpose(parts, 1, 0)
        # (N, 4, 3, 64, 64)
        return parts


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * 64 * 64, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.adv_layer = nn.Linear(64, 1)
        self.classify_block = nn.Linear(64, 4)

    def forward(self, img):
        img_flat = img.view(-1, 3 * 64 * 64)
        out = self.model(img_flat)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.classify_block(out)
        # (N , 2)
        return validity, label


class Parts_Discriminator(nn.Module):
    def __init__(self):
        super(Parts_Discriminator, self).__init__()
        self.model = nn.ModuleList([Discriminator()
                                    for _ in range(4)])

    def forward(self, parts):
        eye1 = self.model[0](parts[:, 0])
        eye2 = self.model[1](parts[:, 1])
        nose = self.model[2](parts[:, 2])
        mouth = self.model[3](parts[:, 3])

        return eye1, eye2, nose, mouth



