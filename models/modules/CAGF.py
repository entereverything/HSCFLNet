import torch
import torch.nn as nn

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU()
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        out = identity * a_w * a_h
        return out

class CAGF(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAGF, self).__init__()
        self.num_heads = num_heads
        self.conv_branch = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), padding=1, bias=bias),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), padding=1, bias=bias),
        )
        self.coord_att = CoordAtt(dim)
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.final_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.drop = DropPath(0.3)

    def forward(self, x):
        residual = x
        x_3d = x.unsqueeze(2)
        out_conv = self.conv_branch(x_3d).squeeze(2)
        out_attn = self.coord_att(x)
        fusion = torch.cat([out_conv, out_attn], dim=1)
        gate = self.fusion_gate(fusion)
        out = gate * out_conv + (1 - gate) * out_attn
        out = self.final_proj(out)
        out = self.drop(out)
        return residual + out