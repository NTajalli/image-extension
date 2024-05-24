import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import os

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm([in_dim, in_dim // 8, in_dim // 8])  # Using LayerNorm

        # Reinitialize weights
        nn.init.kaiming_normal_(self.query_conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.key_conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.value_conv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.nn.functional.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x

        # Apply normalization
        out = self.norm(out)

        # Save attention maps for debugging
        if not os.path.exists("debug/attention"):
            os.makedirs("debug/attention")
        for i in range(min(2, batch_size)):  # Save attention maps for first 2 images
            plt.imshow(attention[i].cpu().detach().numpy())
            plt.colorbar()
            plt.title(f'Attention Map - Image {i}')
            plt.savefig(f'debug/attention/attention_map_image_{i}.png')
            plt.close()

        return out


class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SelfAttention(256),
            nn.Dropout(0.5)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SelfAttention(256)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        # Save encoder feature maps for debugging
        if not os.path.exists("debug/features"):
            os.makedirs("debug/features")
        plt.imshow(enc1[0, 0].cpu().detach().numpy(), cmap='gray')
        plt.colorbar()
        plt.title('Encoder Block 1 Feature Map')
        plt.savefig('debug/features/enc1_feature_map.png')
        plt.close()

        plt.imshow(enc3[0, 0].cpu().detach().numpy(), cmap='gray')
        plt.colorbar()
        plt.title('Encoder Block 3 Feature Map')
        plt.savefig('debug/features/enc3_feature_map.png')
        plt.close()

        # Decoder with skip connections
        dec1 = self.dec1(enc5)
        dec1 = torch.cat((dec1, enc4), dim=1)  # Skip connection from enc4

        dec2 = self.dec2(dec1)
        dec2 = torch.cat((dec2, enc3), dim=1)  # Skip connection from enc3

        dec3 = self.dec3(dec2)
        dec3 = torch.cat((dec3, enc2), dim=1)  # Skip connection from enc2

        dec4 = self.dec4(dec3)
        dec4 = torch.cat((dec4, enc1), dim=1)  # Skip connection from enc1

        dec5 = self.dec5(dec4)

        return dec5

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(256),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg = nn.Sequential(*list(vgg.children())[:35]).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.criterion = nn.MSELoss()

    def forward(self, fake, real):
        fake_features = self.vgg(fake)
        real_features = self.vgg(real)
        return self.criterion(fake_features, real_features)
