import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
import torch
import torch.nn as nn

from main.resnet import r2plus1d_18


class R2plus1d18KeepTemp(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()

        self.model = r2plus1d_18(pretrained=pretrained)

        self.model.layer2[0].conv1[0][3] = nn.Conv3d(230, 128, kernel_size=(3, 1, 1),
                                                     stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.model.layer2[0].downsample = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(1, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.model.layer3[0].conv1[0][3] = nn.Conv3d(460, 256, kernel_size=(3, 1, 1),
                                                     stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.model.layer3[0].downsample = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(1, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.model.layer4[0].conv1[0][3] = nn.Conv3d(921, 512, kernel_size=(3, 1, 1),
                                                     stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.model.layer4[0].downsample = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=(1, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.model.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.model.fc = nn.Identity()

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = self.model(x)
        return x


class VideoOnsetNet(nn.Module):
    # Video Onset detection network
    def __init__(self, pretrained=False):
        super(VideoOnsetNet, self).__init__()
        self.net = R2plus1d18KeepTemp(pretrained=pretrained)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.transpose(-1, -2)
        x = self.fc(x)
        x = x.squeeze(-1)

        return x


if __name__ == '__main__':
    model = VideoOnsetNet(False).cuda()
    rand_input = torch.randn((1, 3, 30, 112, 112)).cuda()
    out = model(rand_input)
    print(out)
