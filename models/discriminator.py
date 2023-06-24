import torch
import torch.nn as nn
import torchvision.models as models

class ResNetDiscriminator(nn.Module):
    def __init__(self, channels=3):
        super(ResNetDiscriminator, self).__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True))

            # # State (512x8x8)
            # nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(16),
            # nn.LeakyReLU(0.2, inplace=True))
            # outptut of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=4, stride=1, padding=0),
            # Output 1
            nn.Sigmoid())

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384 features
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)

# class ResNetDiscriminator(nn.Module):
#     def __init__(self, channels=3):
#         super(ResNetDiscriminator, self).__init__()
#         # Filters [256, 512, 1024]
#         # Input_dim = channels (Cx64x64)
#         # Output_dim = 1
#         self.main_module = nn.Sequential(
#             # Image (Cx32x32)
#             nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # State (256x16x16)
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # State (512x8x8)
#             nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(1024),
#             nn.LeakyReLU(0.2, inplace=True))
#             # outptut of main module --> State (1024x4x4)
#
#         self.output = nn.Sequential(
#             nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0),
#             # Output 1
#             nn.Sigmoid())
#
#     def forward(self, x):
#         x = self.main_module(x)
#         return self.output(x)
