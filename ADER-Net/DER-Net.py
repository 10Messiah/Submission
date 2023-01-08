import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from fvcore.nn import flop_count_table, flop_count_str
from fvcore.nn import FlopCountAnalysis

class Fusion_module(nn.Module):
    def __init__(self,rgb_channels,sal_channels):
        super(Fusion_module, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv3d(rgb_channels+sal_channels,rgb_channels,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
            nn.BatchNorm3d(rgb_channels),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv3d(rgb_channels, rgb_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=(1, 1, 1)),
            nn.BatchNorm3d(rgb_channels),
            nn.Sigmoid()
        )

        self.conv2 = nn.Sequential(
            # nn.Conv3d(rgb_channels, rgb_channels, kernel_size=(5, 5, 5), stride=(1, 1, 1),padding=(2, 2, 2)),
            nn.Conv3d(rgb_channels, rgb_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1),padding=(2, 2, 2),dilation=2),
            nn.BatchNorm3d(rgb_channels),
            nn.Sigmoid()
        )


    def forward(self, f_rgb,f_sal):
        cat_feature = torch.cat((f_rgb,f_sal),dim=1)
        x1 = self.conv0(cat_feature)
        x2 = self.conv1(x1)
        x3 = self.conv2(x1)

        x4 = x2*x3
        x5 = f_rgb+x4
        return x5

class Weighting_Module(nn.Module):
    def __init__(self):
        super(Weighting_Module,self).__init__()

    def cc(self,s_map, gt):
        s_map_norm = (s_map - torch.mean(s_map)) / torch.std(s_map)
        gt_norm = (gt - torch.mean(gt)) / torch.std(gt)
        a = s_map_norm
        b = gt_norm
        r = torch.sum(a * b) / torch.sqrt(torch.sum(a * a) * torch.sum(b * b))
        return r

    def sim(self,s_map, gt):

        s_map_norm = (s_map - torch.min(s_map)) / (torch.max(s_map) - torch.min(s_map))
        s_map_norm = s_map_norm / torch.sum(s_map_norm)
        gt_norm = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt))
        gt_norm = gt_norm / torch.sum(gt_norm)
        t1 = (torch.minimum(s_map_norm, gt_norm))
        t2 = torch.sum(t1)

        return t2

    def function_F(self,s_map, gt,alpha):
        score_total = torch.zeros(s_map.size(0))
        for i in range(s_map.size(0)):
            score_total[i] = alpha*self.cc(s_map[i],gt[i])+(1-alpha)*self.sim(s_map[i],gt[i])
        return score_total

    def function_G(self, score):
        for i in range(score.size(0)):
            score[i] = 1-score[i]
            score[i] = score[i]/torch.max(score[i])
            score[i] = torch.exp(score[i])
        return score

    def forward(self, x):
        x1 = torch.unbind(x, dim=2)
        x_mean = torch.mean(x,dim=2)
        x_socre = torch.zeros((x.size(2),x.size(0))).cuda()
        for i in range(16):
            x_socre[i] = (self.function_F(x1[i],x_mean,0.3))
        x_socre = x_socre.permute((1,0))
        x_socre = self.function_G(x_socre)
        x_socre = x_socre.unsqueeze(1)
        x_socre = x_socre.unsqueeze(3)
        x_socre = x_socre.unsqueeze(4)
        x_res = x*x_socre

        return x_res


class Fusion_3D_with_weights(nn.Module):


    def __init__(self):
        super(Fusion_3D_with_weights, self).__init__()
        pretrained_cnn = torchvision.models.video.r3d_18(pretrained=True, progress=True)
        cnn_layers_1 = list(pretrained_cnn.children())[:3]
        cnn_layers_2 = list(pretrained_cnn.children())[3:-2]

        self.rgb_resnet_1 = nn.Sequential(*cnn_layers_1)

        self.weight_module = Weighting_Module()
        self.sal_conv0 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.sal_conv1_0 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.sal_conv1_1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.sal_conv2_0 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.sal_conv2_1 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )

        self.fusion_0 = Fusion_module(128,128)


        self.resnet_2 = nn.Sequential(*cnn_layers_2)
        self.conv = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.bn = nn.BatchNorm3d(512)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.liner_1 = nn.Linear(512, 51)
        self.liner_2 = nn.Linear(51, 6)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, images,saliency):
        # for i in range(saliency.size(0)):
        #     for j in range(saliency.size(2)):
        #         print(torch.max(saliency[i][0][j]),torch.mean(saliency[i][0][j]),torch.min(saliency[i][0][j]))
        #     print()
        saliency = self.weight_module(saliency)
        # print(saliency.size())
        # for i in range(saliency.size(0)):
        #     for j in range(saliency.size(2)):
        #         print(torch.max(saliency[i][0][j]),torch.mean(saliency[i][0][j]),torch.min(saliency[i][0][j]))
        #     print()
        x1 = self.sal_conv0(saliency)
        x2 = self.sal_conv1_0(x1)
        x3 = self.sal_conv1_1(x2)
        x4 = self.sal_conv2_0(x3)
        x5 = self.sal_conv2_1(x4)

        z1 = self.rgb_resnet_1(images)

        y2 = self.fusion_0(z1,x5)
        y3 = self.resnet_2(y2)
        y4 = self.conv(y3)
        y5 = self.relu(self.bn(y4))
        y6 = self.pool(y5)
        y7 = y6.view((-1, 512))

        y8 = self.relu(self.liner_1(y7))
        y9 = self.dropout(y8)
        res = self.liner_2(y9)

        return res


#
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = torch.randn(2, 3, 16, 192, 320).cuda()
    # BCTHW
    saliency = torch.randn(2, 1, 16, 192, 320).cuda()
    net = Fusion_3D_with_weights().to(device)
    # net2 = Weighting_Module().to(device)
    out = net(images,saliency)
    print(out.shape)
    print(flop_count_table(FlopCountAnalysis(net,(images,saliency))))




