import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
from PIL import Image
import math
import numpy as np
from scipy import ndimage
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from scipy.stats import mode


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet2, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

        self.P4 = double_conv(512, 512)
        self.P3 = double_conv(256, 256)
        self.P2 = double_conv(128, 128)
        self.P1 = double_conv(64, 64)

        self.up1 = up(1024, 256, bilinear)
        self.up2 = up(512, 128, bilinear)
        self.up3 = up(256, 64, bilinear)
        self.up4 = up(128, 64, bilinear)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, self.P4(x4))
        x = self.up2(x, self.P3(x3))
        x = self.up3(x, self.P2(x2))
        x = self.up4(x, self.P1(x1))
        x = self.outc(x)
        return torch.sigmoid(x)


class Rotation:
    def __init__(self, device):
        super(Rotation, self).__init__()
        self.device = torch.device(device)
        self.unet = UNet2(3, 1)
        self.unet.load_state_dict(torch.load(
            '4.pth', map_location=device))
        self.unet.to(device)
        self.rotate = models.vgg19()
        self.rotate.classifier[6] = nn.Linear(4096, 2)
        self.rotate.load_state_dict(torch.load('2.pth'))
        self.rotate.to(device)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    def __call__(self, card, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = Image.fromarray(img)
        img = self.transform(img)
        img = img.unsqueeze(0).to(self.device)
        pred = self.unet(img)
        pred = (pred > 0.5).float()[0]
        pred = pred.squeeze(0)
        gray_unet = pred.detach().cpu().numpy()
        gray_unet = np.uint8(gray_unet)*255
        h, w = card.shape[:2]
        gray_unet = cv2.resize(gray_unet, (w, h))
        gray_unet = cv2.Canny(gray_unet, 100, 100, apertureSize=3)
        lines = cv2.HoughLinesP(gray_unet, .5, math.pi / 180.0,
                                50, minLineLength=(w+h)/20, maxLineGap=5)
        max_len, x1_last, x2_last, y1_last, y2_last = 0, 0, 10, 0, 0
        if lines is None:
            edges = canny(gray)
            tested_angles = np.deg2rad(np.arange(0.1, 180.0))
            h, theta, d = hough_line(edges, theta=tested_angles)
            angles = hough_line_peaks(h, theta, d)[1]
            most_common_angle = mode(np.around(angles, decimals=2))[0]
            angle = int(np.rad2deg(most_common_angle - np.pi / 2))
            card = ndimage.rotate(card, angle)
        else:
            for [[x1, y1, x2, y2]] in lines:
                now = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)
                if now > max_len:
                    max_len = now
                    x1_last, x2_last, y1_last, y2_last = x1, x2, y1, y2
            angle = math.degrees(math.atan2(
                y2_last - y1_last, x2_last - x1_last))
            card = ndimage.rotate(card, angle)
        r = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
        r = Image.fromarray(r)
        r = self.transform(r)
        r = r.unsqueeze(0).to(self.device)
        _, pred = torch.max(self.rotate(r), 1)
        pred = pred.item()
        if pred is 1:
            card = ndimage.rotate(card, 180)
        return card
