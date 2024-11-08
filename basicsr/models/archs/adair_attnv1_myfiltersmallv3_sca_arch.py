## AdaIR: Adaptive All-in-One Image Restoration via Frequency Mining and Modulation
## Yuning Cui, Syed Waqas Zamir, Salman Khan, Alois Knoll, Mubarak Shah, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2403.14614


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

import torchvision
from einops import rearrange


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class Attention_v1(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_v1, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.project_in =nn.Conv2d(dim,2*dim,kernel_size=1,bias=bias)

        self.qkv_x1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv_x1 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.qkv_x2 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv_x2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = x.chunk(2, dim=1)

        b, c, h, w = x1.shape

        qkv1 = self.qkv_dwconv_x1(self.qkv_x1(x1))
        q1, k1, v1 = qkv1.chunk(3, dim=1)

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v1 = rearrange(v1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.temperature
        attn1 = attn1.softmax(dim=-1)

        out1 = (attn1 @ v1)

        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        qkv2 = self.qkv_dwconv_x2(self.qkv_x2(x2))
        q2, k2, v2 = qkv2.chunk(3, dim=1)

        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        attn2 = (q2 @ k2.transpose(-2, -1)) * self.temperature
        attn2 = attn2.softmax(dim=-1)

        out2 = (attn2 @ v2)

        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        softmax_out1 = out1.softmax(dim=-1)
        softmax_out2 = out2.softmax(dim=-1)

        out = softmax_out1 @ out2 + softmax_out2 @ out1

        out = self.project_out(out)
        return out

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
###################################################################
##filter
def Egde(size=3, channel=1, scale=1e-3):
    if size == 3:
        param = torch.ones((channel, 1, 3, 3), dtype=torch.float32) * (-1)
        for i in range(channel):
            param[i][0][1][1] = 8
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 5:
        param = torch.ones((channel, 1, 5, 5), dtype=torch.float32) * (-1)
        for i in range(channel):
            param[i][0][1][2] = 2
            param[i][0][2][1] = 2
            param[i][0][2][2] = 4
            param[i][0][2][3] = 2
            param[i][0][3][2] = 2
        param = nn.Parameter(data=param * scale, requires_grad=False)

    else:
        raise NotImplementedError

    return param


def Sobel(size=3, channel=1, scale=1e-3, direction='x'):
    if size == 3:
        param = torch.zeros((channel, 1, 3, 3), dtype=torch.float32)
        for i in range(channel):
            param[i][0][0][0] = param[i][0][2][0] = 1
            param[i][0][0][2] = param[i][0][2][2] = -1
            param[i][0][1][0] = 2
            param[i][0][1][2] = -2
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 5:
        param = torch.zeros((channel, 1, 5, 5), dtype=torch.float32)
        for i in range(channel):
            param[i][0][0][0] = param[i][0][4][0] = 1
            param[i][0][0][1] = param[i][0][4][1] = 2
            param[i][0][0][3] = param[i][0][4][3] = -2
            param[i][0][0][4] = param[i][0][4][4] = -1

            param[i][0][1][0] = param[i][0][3][0] = 4
            param[i][0][1][1] = param[i][0][3][1] = 8
            param[i][0][1][3] = param[i][0][3][3] = -8
            param[i][0][1][4] = param[i][0][3][4] = -4

            param[i][0][2][0] = 6
            param[i][0][2][1] = 12
            param[i][0][2][3] = -12
            param[i][0][2][4] = -6
        param = nn.Parameter(data=param * scale, requires_grad=False)

    else:
        raise NotImplementedError

    if direction == 'x':
        return param
    else:
        return param.transpose(3, 2)


def Sobel_xy(size=3, channel=1, scale=1e-3, direction='xy'):
    param = torch.zeros((channel, 1, 3, 3), dtype=torch.float32)
    if size == 3 and direction == 'xy':
        for i in range(channel):
            param[i][0][0][1] = 1
            param[i][0][0][2] = 2
            param[i][0][1][0] = -1
            param[i][0][1][2] = 1
            param[i][0][2][0] = -2
            param[i][0][2][1] = -1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 3 and direction == 'yx':
        for i in range(channel):
            param[i][0][0][0] = -2
            param[i][0][0][1] = -1
            param[i][0][1][0] = -1
            param[i][0][1][2] = 1
            param[i][0][2][1] = 1
            param[i][0][2][2] = 2
        param = nn.Parameter(data=param * scale, requires_grad=False)

    else:
        raise NotImplementedError

    return param


def Roberts(size=3, channel=1, scale=1e-3, direction='x'):
    if size == 3 and direction == 'x':
        param = torch.zeros((channel, 1, 3, 3), dtype=torch.float32)
        for i in range(channel):
            param[i][0][0][0] = 1
            param[i][0][1][1] = -1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 3 and direction == 'y':
        param = torch.zeros((channel, 1, 3, 3), dtype=torch.float32)
        for i in range(channel):
            param[i][0][0][1] = 1
            param[i][0][1][0] = -1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 2 and direction == 'x':
        param = torch.zeros((channel, 1, 2, 2), dtype=torch.float32)
        for i in range(channel):
            param[i][0][0][0] = 1
            param[i][0][1][1] = -1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 2 and direction == 'y':
        param = torch.zeros((channel, 1, 2, 2), dtype=torch.float32)
        for i in range(channel):
            param[i][0][0][1] = 1
            param[i][0][1][0] = -1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    else:
        raise NotImplementedError

    return param


def Prewitt(size=3, channel=1, scale=1e-3, direction='x'):
    param = torch.zeros((channel, 1, 3, 3), dtype=torch.float32)
    if size == 3 and direction == 'y':
        for i in range(channel):
            param[i][0][0][0] = -1
            param[i][0][1][0] = -1
            param[i][0][2][0] = -1
            param[i][0][0][2] = 1
            param[i][0][1][2] = 1
            param[i][0][2][2] = 1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 3 and direction == 'x':
        for i in range(channel):
            param[i][0][0][0] = -1
            param[i][0][0][1] = -1
            param[i][0][0][2] = -1
            param[i][0][2][0] = 1
            param[i][0][2][1] = 1
            param[i][0][2][2] = 1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 3 and direction == 'xy':
        for i in range(channel):
            param[i][0][0][1] = 1
            param[i][0][0][2] = 1
            param[i][0][1][0] = -1
            param[i][0][1][2] = 1
            param[i][0][2][0] = -1
            param[i][0][2][1] = -1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    elif size == 3 and direction == 'yx':
        for i in range(channel):
            param[i][0][0][0] = -1
            param[i][0][0][1] = -1
            param[i][0][1][0] = -1
            param[i][0][1][2] = 1
            param[i][0][2][1] = 1
            param[i][0][2][2] = 1
        param = nn.Parameter(data=param * scale, requires_grad=False)

    else:
        raise NotImplementedError

    return param


def Laplacian(channel=1, scale=1e-3, type=1):
    param = torch.ones((channel, 1, 3, 3), dtype=torch.float32)
    if type == 1:
        for i in range(channel):
            param[i][0][0][0] = 0
            param[i][0][0][2] = 0
            param[i][0][1][1] = -4
            param[i][0][2][0] = 0
            param[i][0][2][2] = 0
        param = nn.Parameter(data=param * scale, requires_grad=False)
    else:
        for i in range(channel):
            param[i][0][1][1] = -4
        param = nn.Parameter(data=param * scale, requires_grad=False)
    return param


def HighPass(x, kernel_size=15, sigma=5):
    filter2 = torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    gauss = filter2(x)
    return x - gauss

class Filters(nn.Module):
    def __init__(self, dim):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')

        self.dim = dim
        self.Sobel_x = Sobel(channel=dim, direction='x').to(device)
        self.Sobel_y = Sobel(channel=dim, direction='y').to(device)
        self.Laplation = Laplacian(channel=dim).to(device)
        self.Edge = Egde(channel=dim).to(device)
        self.Roberts_x = Roberts(channel=dim, direction='x').to(device)
        self.Roberts_y = Roberts(channel=dim, direction='y').to(device)
        self.Sobel_xy = Sobel_xy(channel=dim, direction='xy').to(device)
        self.Sobel_yx = Sobel_xy(channel=dim, direction='yx').to(device)
        self.Prewitt_x = Prewitt(channel=dim, direction='x').to(device)
        self.Prewitt_y = Prewitt(channel=dim, direction='y').to(device)
        self.Prewitt_xy = Prewitt(channel=dim, direction='xy').to(device)
        self.Prewitt_yx = Prewitt(channel=dim, direction='yx').to(device)
        self.alpha = nn.Parameter(torch.ones_like(torch.FloatTensor(13)).to(device).requires_grad_())
        self.beta = nn.Parameter(torch.zeros_like(torch.FloatTensor(1)).to(device).requires_grad_())

        # self.conv3_single = nn.Conv2d(dim,dim,kernel_size=3, padding=1, stride=1,groups=dim,bias=False)
        # self.beta1 = nn.Parameter(torch.zeros_like(torch.FloatTensor(1)).to(device).requires_grad_())
        # self.conv1 = nn.Conv2d(dim,dim,kernel_size=1, padding=0, stride=1, groups=dim,bias=False)
        # self.conv3 = nn.Conv2d(dim,dim,kernel_size=3, padding=1, stride=1,groups=dim,bias=False)
        # self.beta2 = nn.Parameter(torch.zeros_like(torch.FloatTensor(1)).to(device).requires_grad_())


    def forward(self, x):

        Sobel_x = F.conv2d(input=x, weight=self.Sobel_x, stride=1, groups=self.dim, padding=1) * self.alpha[0]
        Sobel_y = F.conv2d(input=x, weight=self.Sobel_y, stride=1, groups=self.dim, padding=1) * self.alpha[1]
        Laplation = F.conv2d(input=x, weight=self.Laplation, stride=1, groups=self.dim, padding=1) * self.alpha[2]
        Egde = F.conv2d(input=x, weight=self.Edge, stride=1, groups=self.dim, padding=1) * self.alpha[3]
        Sobel_xy = F.conv2d(input=x, weight=self.Sobel_xy, stride=1, groups=self.dim, padding=1) * self.alpha[4]
        Sobel_yx = F.conv2d(input=x, weight=self.Sobel_yx, stride=1, groups=self.dim, padding=1) * self.alpha[5]
        Roberts_x = F.conv2d(input=x, weight=self.Roberts_x, stride=1, groups=self.dim, padding=1) * self.alpha[6]
        Roberts_y = F.conv2d(input=x, weight=self.Roberts_y, stride=1, groups=self.dim, padding=1) * self.alpha[7]
        high_pass = HighPass(x) * self.alpha[8]
        Prewitt_x = F.conv2d(input=x, weight=self.Prewitt_x, stride=1, groups=self.dim, padding=1)* self.alpha[9]
        Prewitt_y = F.conv2d(input=x, weight=self.Prewitt_y, stride=1, groups=self.dim, padding=1)* self.alpha[10]
        Prewitt_xy = F.conv2d(input=x, weight=self.Prewitt_xy, stride=1, groups=self.dim, padding=1) * self.alpha[11]
        Prewitt_yx = F.conv2d(input=x, weight=self.Prewitt_yx, stride=1, groups=self.dim, padding=1) * self.alpha[12]
        return (Sobel_x + Sobel_y + Laplation + Egde + x * self.beta[0] +
                Sobel_xy + Sobel_yx + Roberts_x + Roberts_y + high_pass + Prewitt_x + Prewitt_y +Prewitt_xy + Prewitt_yx)


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.filter = Filters(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.filter(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))

        return x

class AttnBlock_v1(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(AttnBlock_v1, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.filter = Filters(dim)
        self.attn = Attention_v1(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.filter(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))

        return x

###################  Simplified Channel Attention #############################
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class Simplified_Channel_Attention(nn.Module):
    def __init__(self, dim, DW_Expand=2):
        super().__init__()
        dw_channel = dim * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=dim, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        return x

##########################################################################
## Channel-Wise Cross Attention (CA)
class Chanel_Cross_Attention(nn.Module):
    def __init__(self, dim, num_head, bias):
        super(Chanel_Cross_Attention, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # x -> q, y -> kv
        assert x.shape == y.shape, 'The shape of feature maps from image and features are not equal!'

        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## H-L Unit
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()

        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        max = torch.max(x, 1, keepdim=True)[0]
        mean = torch.mean(x, 1, keepdim=True)
        scale = torch.cat((max, mean), dim=1)
        scale = self.spatial(scale)
        scale = F.sigmoid(scale)
        return scale


##########################################################################
## L-H Unit
class ChannelGate(nn.Module):
    def __init__(self, dim):
        super(ChannelGate, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.max = nn.AdaptiveMaxPool2d((1, 1))

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(dim // 16, dim, 1, bias=False)
        )

    def forward(self, x):
        avg = self.mlp(self.avg(x))
        max = self.mlp(self.max(x))

        scale = avg + max
        scale = F.sigmoid(scale)
        return scale


##########################################################################
## Frequency Modulation Module (FMoM)
class FreRefine(nn.Module):
    def __init__(self, dim):
        super(FreRefine, self).__init__()

        self.SpatialGate = SpatialGate()
        self.ChannelGate = ChannelGate(dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, low, high):
        spatial_weight = self.SpatialGate(high)
        channel_weight = self.ChannelGate(low)
        high = high * channel_weight
        low = low * spatial_weight

        out = low + high
        out = self.proj(out)
        return out


##########################################################################
## Adaptive Frequency Learning Block (AFLB)
##########################################################################
## Adaptive Frequency Learning Block (AFLB)
class FreModule(nn.Module):
    def __init__(self, dim, num_heads, bias, in_dim=3):
        super(FreModule, self).__init__()

        self.conv = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.score_gen = nn.Conv2d(2, 2, 7, padding=3)

        self.para1 = nn.Parameter(torch.zeros(dim, 1, 1))
        self.para2 = nn.Parameter(torch.ones(dim, 1, 1))

        self.sca_l = Simplified_Channel_Attention(dim)
        self.sca_h = Simplified_Channel_Attention(dim)
        self.sca_agg = Simplified_Channel_Attention(dim)

        self.conv1_l = nn.Conv2d(in_channels=2 * dim, out_channels=dim, kernel_size=1, bias=False)
        self.conv1_h = nn.Conv2d(in_channels=2 * dim, out_channels=dim, kernel_size=1, bias=False)
        self.conv1_agg = nn.Conv2d(in_channels=2 * dim, out_channels=dim, kernel_size=1, bias=False)

        self.frequency_refine = FreRefine(dim)

        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 8, 2, 1, bias=False),
        )

    def forward(self, x, y):
        _, _, H, W = y.size()
        x = F.interpolate(x, (H, W), mode='bilinear')

        high_feature, low_feature = self.fft(x)


        high_feature = self.sca_l(high_feature)

        high_feature = self.conv1_l(torch.cat([high_feature, y], 1))

        low_feature = self.sca_h(low_feature)
        low_feature = self.conv1_h(torch.cat([low_feature, y], 1))

        agg = self.frequency_refine(low_feature, high_feature)

        out = self.conv1_agg(torch.cat([agg,self.sca_agg(y)], 1))

        return out * self.para1 + y * self.para2

    def shift(self, x):
        '''shift FFT feature map to center'''
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h / 2), int(w / 2)), dims=(2, 3))

    def unshift(self, x):
        """converse to shift operation"""
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(-int(h / 2), -int(w / 2)), dims=(2, 3))

    def fft(self, x, n=128):
        """obtain high/low-frequency features from input"""
        x = self.conv1(x)
        mask = torch.zeros(x.shape).to(x.device)
        h, w = x.shape[-2:]

        threshold = F.adaptive_avg_pool2d(x, 1)
        threshold = self.rate_conv(threshold).sigmoid()

        for i in range(mask.shape[0]):
            h_ = (h // n * threshold[i, 0, :, :]).int()
            w_ = (w // n * threshold[i, 1, :, :]).int()

            mask[i, :, h // 2 - h_:h // 2 + h_, w // 2 - w_:w // 2 + w_] = 1

        fft = torch.fft.fft2(x, norm='forward', dim=(-2, -1))
        fft = self.shift(fft)

        fft_high = fft * (1 - mask)

        high = self.unshift(fft_high)
        high = torch.fft.ifft2(high, norm='forward', dim=(-2, -1))
        high = torch.abs(high)

        fft_low = fft * mask

        low = self.unshift(fft_low)
        low = torch.fft.ifft2(low, norm='forward', dim=(-2, -1))
        low = torch.abs(low)

        return high, low


##########################################################################
##---------- AdaIR -----------------------

class AdaIR_s3(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 decoder=True,
                 dual_pixel_task=False
                 ):

        super(AdaIR_s3, self).__init__()
        self.dual_pixel_task = dual_pixel_task
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.decoder = decoder

        if self.decoder:
            self.fre1 = FreModule(dim * 2 ** 3, num_heads=heads[2], bias=bias)
            self.fre2 = FreModule(dim * 2 ** 2, num_heads=heads[2], bias=bias)
            self.fre3 = FreModule(dim * 2 ** 1, num_heads=heads[2], bias=bias)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            AttnBlock_v1(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, noise_emb=None):

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        if self.decoder:
            latent = self.fre1(inp_img, latent)

        inp_dec_level3 = self.up4_3(latent)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        if self.decoder:
            out_dec_level3 = self.fre2(inp_img, out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        if self.decoder:
            out_dec_level2 = self.fre3(inp_img, out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

#----------test----------
if __name__ == '__main__':
    #输入（1，3，x，y） N C H W, 输出也是一样的
    #pretrain_network_g: ./Deraining/pretrained_models/net_g_252000_attnv1.pth
    #mini_batch_sizes: [4,2,2,1,1,1]
    x = torch.randn(1, 3, 64, 64).cuda()
    model =AdaIR_s3().cuda()
    # print(model)
    y = model(x)
    print(y.shape)
