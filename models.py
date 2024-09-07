from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self,num_filters = 256):
        super(ResidualBlock,self).__init__()
        self.conv1 = nn.Conv2d(num_filters,num_filters,kernel_size=3,padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filters,num_filters,kernel_size=3,padding=1)

    def forward(self,x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + residual

class mtf_sr(nn.Module):
    def __init__(self,num_filters = 256,num_ResdualBlocks=32,scale=2):
        super(mtf_sr,self).__init__()
        self.conv1 = nn.Conv2d(3,num_filters,kernel_size=3,padding=1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_ResdualBlocks)]
        )
        self.conv2 = nn.Conv2d(num_filters,num_filters,kernel_size=3,padding=1)
        self.subpixel_conv = nn.Conv2d(num_filters, num_filters * (scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.conv3 = nn.Conv2d(num_filters,1,kernel_size=3,padding=1)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.res_blocks(x)
        x = self.conv2(x)
        x = self.subpixel_conv(x)
        x = self.pixel_shuffle(x)
        x = self.conv3(x)
        return x