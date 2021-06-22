import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from quant_ultra import *


class MobileNetV1Qua(nn.Module):
    def __init__(self,num_classes=1000):
        super(MobileNetV1Qua, self).__init__()
        W_BIT = 4
        A_BIT = 4
        AVGPOOL_A_BIT = 5
        conv2d_q = conv2d_Q_fn(W_BIT)
        linear_q = linear_Q_fn(W_BIT)
        

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                conv2d_q(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                #nn.ReLU(inplace=True),
                activation_quantize_fn(A_BIT)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                conv2d_q(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                #nn.ReLU(inplace=True),
                activation_quantize_fn(A_BIT),
    
                conv2d_q(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                #nn.ReLU(inplace=True),
                activation_quantize_fn(A_BIT)
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            #conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            #conv_dw(512, 512, 1),
            #conv_dw(512, 512, 1),
            #conv_dw(512, 512, 1),
            #conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
           # conv_dw(1024, 1024, 1),
            # nn.AvgPool2d(7),
        )
        self.avg = nn.AvgPool2d(7)
        self.act_q = activation_quantize_fn_avgpool(AVGPOOL_A_BIT)
        self.fc = linear_q(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.avg(x)
        x = x*49/32
        x = self.act_q(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

    def get_15out(self,x):
        x = self.model(x)
        x = self.avg(x)
        x = x*49/32
        x = self.act_q(x)
        return x

    def get_14out(self,x):
        x = self.model(x)
        return x
if __name__ == '__main__':
    a = torch.randn(8, 3, 224, 224)
    net = MobileNetV1Qua()
    res = net(a)
    print(res.shape)
