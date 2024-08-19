import torch
import torch.nn as nn
import torch.nn.functional as F

class FFTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(FFTConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class UNetFFT(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(UNetFFT, self).__init__()

        self.encoder1 = self.double_conv(in_channels, 64, modes1, modes2)
        self.encoder2 = self.double_conv(64, 128, modes1, modes2)
        self.encoder3 = self.double_conv(128, 256, modes1, modes2)
        self.encoder4 = self.double_conv(256, 512, modes1, modes2)
        
        self.pool = nn.MaxPool2d(2)
        
        self.middle = self.double_conv(512, 1024, modes1, modes2)
        
        self.upconv4 = self.upconv(1024, 512)
        self.decoder4 = self.double_conv(1024, 512, modes1, modes2)
        
        self.upconv3 = self.upconv(512, 256)
        self.decoder3 = self.double_conv(512, 256, modes1, modes2)
        
        self.upconv2 = self.upconv(256, 128)
        self.decoder2 = self.double_conv(256, 128, modes1, modes2)
        
        self.upconv1 = self.upconv(128, 64)
        self.decoder1 = self.double_conv(128, 64, modes1, modes2)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels, modes1, modes2):
        return nn.Sequential(
            FFTConv2d(in_channels, out_channels, modes1, modes2),
            nn.ReLU(inplace=True),
            FFTConv2d(out_channels, out_channels, modes1, modes2),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # Middle
        middle = self.middle(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(middle)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        out = self.final_conv(dec1)
        return out

# Example usage

if __name__ == '__main__':
    img = torch.randn(8, 3, 256, 256)
    model = UNetFFT(in_channels=3, out_channels=1, modes1=8, modes2=8)
    print(model)

    print(model(img))
