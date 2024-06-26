import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channles, out_channles):
        super(DoubleConv, self).__init__()
        self.conv =  nn.Sequential(
            nn.Conv2d(in_channles, out_channles, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channles),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channles, out_channles, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channles),
            nn.ReLU(inplace=True),


        )
    
    def forward(self, x):
        return self.conv(x)
    

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=8, features=[64, 128, 256, 512],
    ):
        super(Unet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        #for the down part of the unet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #for the up part 
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(
                feature*2, feature, kernel_size=2, stride=2
            ))
            self.ups.append(DoubleConv(feature*2 , feature))


        self.lastlayer = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x= down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x= self.lastlayer(x)

        skip_connections = skip_connections[::-1] 
        # reverse the skip connectio list

        for index in range(0, len(self.ups), 2):
            x= self.ups[index](x)
            skip_connection = skip_connections[index//2]

            if x.shape != skip_connection.shape:
                x = TF.resize( x, size = skip_connection.shape[2:])


            concatenate = torch.cat((skip_connection, x), dim=1)
            x= self.ups[index+1](concatenate)
        
        return F.softmax(self.final_conv(x), dim=1)


def test():
    x= torch.randn((3, 3, 512, 512))
    model = UNET(in_channels=3, out_channels=8)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()