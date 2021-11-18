import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import transforms


def conv3x3(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, 1, 1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True))

class CDNN(nn.Module):
    def __init__(self):
        n, m = 8, 3

        super(CDNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.convd1 = conv3x3(1 * m, 1 * n)
        self.convd2 = conv3x3(1 * n, 2 * n)
        self.convd3 = conv3x3(2 * n, 4 * n)
        self.convd4 = conv3x3(4 * n, 4 * n)

        self.convu3 = conv3x3(8 * n, 4 * n)
        self.convu2 = conv3x3(6 * n, 2 * n)
        self.convu1 = conv3x3(3 * n, 1 * n)

        self.convu0 = nn.Conv2d(n, 1, 3, 1, 1)


    def forward(self, x):
        x1 = x
        x1 = self.convd1(x1)

        x2 = self.maxpool(x1)
        x2 = self.convd2(x2)

        x3 = self.maxpool(x2)
        x3 = self.convd3(x3)

        x4 = self.maxpool(x3)
        x4 = self.convd4(x4)

        y3 = self.upsample(x4)
        y3 = torch.cat([x3, y3], 1)
        y3 = self.convu3(y3)

        y2 = self.upsample(y3)
        y2 = torch.cat([x2, y2], 1)
        y2 = self.convu2(y2)

        y1 = self.upsample(y2)
        y1 = torch.cat([x1, y1], 1)
        y1 = self.convu1(y1)

        y1 = self.convu0(y1)
        y1 = self.sigmoid(y1)

        return y1

class ADER_Net(nn.Module):


    def __init__(self):
        super(ADER_Net, self).__init__()
        pretrained_cnn = torchvision.models.video.r3d_18(pretrained=True,progress=True)
        cnn_layers_1 = list(pretrained_cnn.children())[:3]
        cnn_layers_2 = list(pretrained_cnn.children())[3:-2]
        self.resnet_1 = nn.Sequential(*cnn_layers_1)
        self.resnet_2 = nn.Sequential(*cnn_layers_2)
        self.sali = CDNN()
        self.conv = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1,2,2),padding=(1, 1, 1))
        self.bn = nn.BatchNorm3d(512)
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.liner_1 = nn.Linear(512,51)
        self.liner_2 = nn.Linear(51,6)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU(inplace=True)

        cnt = 0
        for name, para in ADER_Net.named_parameters(self): #fix the weights of CDNN
            cnt = cnt + 1
            if ((cnt>60)and(cnt<119)) :
                para.requires_grad = False



    def forward(self, x):
        x1 = torch.unbind(x,dim=2)
        x2 = torch.zeros(size=(x.size(2),x.size(0),x.size(3),x.size(4))).cuda()

        for i in range(len(x1)):
            x2[i] = self.sali(x1[i]).permute(1,0,2,3)

        x3 = torch.mean(x2,dim=0)
        x4 = torch.zeros(size=x3.size()).cuda()
        for i in range(x3.size(0)):
            x4[i] = (x3[i]-torch.min(x3[i]))/(torch.max(x3[i]-torch.min(x3[i])))
        x5 = x4.unsqueeze(1)
        x5 = x5.unsqueeze(1)
        x6 = F.interpolate(x5, size=[1,48, 80])


        y1 = self.resnet_1(x)
        y2 =torch.addcmul(input=y1,tensor1=y1,tensor2=x6,value=0.5)
        y3 = self.resnet_2(y2)

        y4 = self.conv(y3)
        y5 = self.relu(self.bn(y4))
        y6 = self.pool(y5)
        y7 = y6.view((-1,512))

        y8 = self.relu(self.liner_1(y7))
        y9 = self.dropout(y8)
        res = self.liner_2(y9)

        return res

#
