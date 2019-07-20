import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils import data
import torch.utils.model_zoo as model_zoo
from torchvision import models


def downsample(xs, scale):
    if scale == 1:
        return xs

    # find new size dividable by 32
    h = xs.size()[2]
    w = xs.size()[3]

    new_h = int(h * scale)
    new_w = int(w * scale)
    new_h = new_h + 32 - new_h % 32
    new_w = new_w + 32 - new_w % 32

    dsize = (new_w, new_h)
    ys = []
    for x in xs:
        x = torch.from_numpy(x)  # c,h,w
        if x.ndim == 3:
            x = np.transpose(x, [1, 2, 0])
            x = cv2.resize(x, dsize=dsize, interpolation=cv2.INTER_LINEAR)
            x = np.transpose(x, [2, 0, 1])
        else:
            x = cv2.resize(x, dsize=dsize, interpolation=cv2.INTER_LINEAR)

        ys.append(torch.unsqueeze(torch.from_numpy(x), dim=0))

    return ys

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1_p = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True)

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/16, 1024
        self.res5 = resnet.layer4  # 1/32, 2048

        # freeze BNs
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f, in_p):
        f = (in_f - Variable(self.mean)) / Variable(self.std)
        p = torch.unsqueeze(in_p, dim=1).float()  # add channel dim

        x = self.conv1(f) + self.conv1_p(p)  # + self.conv1_n(n)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 64
        r3 = self.res3(r2)  # 1/8, 128
        r4 = self.res4(r3)  # 1/16, 256
        r5 = self.res5(r4)  # 1/32, 512

        return r5, r4, r3, r2


class GC(nn.Module):
    def __init__(self, inplanes, planes, kh=7, kw=7):
        super(GC, self).__init__()
        self.conv_l1 = nn.Conv2d(inplanes, 256, kernel_size=(kh, 1),
                                 padding=(int(kh / 2), 0))
        self.conv_l2 = nn.Conv2d(256, planes, kernel_size=(1, kw),
                                 padding=(0, int(kw / 2)))
        self.conv_r1 = nn.Conv2d(inplanes, 256, kernel_size=(1, kw),
                                 padding=(0, int(kw / 2)))
        self.conv_r2 = nn.Conv2d(256, planes, kernel_size=(kh, 1),
                                 padding=(int(kh / 2), 0))

    def forward(self, x):
        x_l = self.conv_l2(self.conv_l1(x))
        x_r = self.conv_r2(self.conv_r1(x))
        x = x_l + x_r
        return x


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.convFS2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convFS3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.scale_factor = scale_factor
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, f, pm):
        s = self.convFS1(f)
        sr = self.ca(s)*s
        sr = self.sa(sr)*sr
        sr = self.convFS2(F.relu(sr))
        sr = self.convFS3(F.relu(sr))
        s = s + sr


        m = s + F.upsample(pm, scale_factor=self.scale_factor, mode='bilinear')

        mr = self.convMM1(F.relu(m))
        mr = self.convMM2(F.relu(mr))
        m = m + mr
        return m

class RefineResidual(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, has_bias=False,
                 has_relu=False, norm_layer=nn.BatchNorm2d, bn_eps=1e-5):
        super(RefineResidual, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                  stride=1, padding=0, dilation=1,
                                  bias=has_bias)
        self.cbr = ConvBnRelu(out_planes, out_planes, ksize, 1,
                              ksize // 2, has_bias=has_bias,
                              norm_layer=norm_layer, bn_eps=bn_eps)
        self.conv_refine = nn.Conv2d(out_planes, out_planes, kernel_size=ksize,
                                     stride=1, padding=ksize // 2, dilation=1,
                                     bias=has_bias)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv_1x1(x)
        t = self.cbr(x)
        t = self.conv_refine(t)
        if self.has_relu:
            return self.relu(t + x)
        return t + x

class Refine1(nn.Module):
    def __init__(self, in_planes, out_planes,
                 reduction=1, norm_layer=nn.BatchNorm2d):
        super(Refine, self).__init__()
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes // reduction, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=True, has_bias=False),
            ConvBnRelu(out_planes // reduction, out_planes, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=False, has_bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # fm = torch.cat([x1, x2], dim=1)
        # fm = self.conv_1x1(x1)
        fm_se = self.channel_attention(x1)
        fm = x1*fm_se
        output = fm + x2
        return output

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        mdim = 256
        self.GC = GC(4096, mdim)  # 1/32 -> 1/32
        self.convG1 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)
        self.convG2 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)
        self.RF4 = Refine(1024, mdim)  # 1/16 -> 1/8
        self.RF3 = Refine(512, mdim)  # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim)  # 1/4 -> 1

        self.pred5 = nn.Conv2d(mdim, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.pred4 = nn.Conv2d(mdim, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.pred3 = nn.Conv2d(mdim, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, r5, x5, r4, r3, r2):

        x = torch.cat((r5, x5), dim=1)

        x = self.GC(x)
        r = self.convG1(F.relu(x))
        r = self.convG2(F.relu(r))
        m5 = x + r  # out: 1/32, 64
        m4 = self.RF4(r4, m5)  # out: 1/16, 64
        # print('m4.size() is:', m4.size())
        m3 = self.RF3(r3, m4)  # out: 1/8, 64
        # print('m3.size() is:', m3.size())
        m2 = self.RF2(r2, m3)  # out: 1/4, 64
        # print('m2.size() is:', m2.size())

        p2 = self.pred2(F.relu(m2))
        p3 = self.pred3(F.relu(m3))
        p4 = self.pred4(F.relu(m4))
        p5 = self.pred5(F.relu(m5))

        p = F.upsample(p2, scale_factor=4, mode='bilinear')

        return p, p2, p3, p4, p5


class RGMP(nn.Module):
    def __init__(self):
        super(RGMP, self).__init__()
        self.Encoder = Encoder()
        self.Decoder = Decoder()


class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")

