import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.cswin import CSWin_64_12211_tiny_224, CSWin_64_24322_small_224, CSWin_96_24322_base_384, \
    CSWin_96_24322_base_224
from models.head.FCN import FCNHead
from models.neck.FPN import FPNNeck
from collections import OrderedDict
from util.common import ScaleInOutput
from models.modules.relukan_conv import  ReLUKANConv2DLayer
from models.modules.CAGF import CAGF

class Conv3Relu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Conv3Relu, self).__init__()
        self.extract = nn.Sequential(nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1),
                                               stride=(stride, stride), bias=False),
                                     nn.BatchNorm2d(out_ch))

    def forward(self, x):
        x = self.extract(x)
        return x

class ChangeDetection(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.inplanes = int(re.sub(r"\D", "", opt.backbone.split("_")[-1]))
        self.dl = opt.dual_label
        self.auxiliary_head = False

        self._create_backbone(opt.backbone)
        self._create_neck(opt.neck)
        self._create_heads(opt.head)
        
        self.contrast_loss = Conv3Relu(512,2)
        self.cv_kan = ReLUKANConv2DLayer(2,2,kernel_size=3,padding=1,dropout=0.2)
    
        self.CAGF1 = CAGF(64,8,False)
        self.CAGF2 = CAGF(128,8,False)
        self.CAGF3 = CAGF(256,8,False)
        self.CAGF4 = CAGF(512,8,False)

        if opt.pretrain.endswith(".pt"):
            self._init_weight(opt.pretrain)
        self._model_summary(opt.input_size)

    def forward(self, xa, xb, tta=False):
        if not tta:
            return self.forward_once(xa, xb)
        else:
            return self.forward_tta(xa, xb)

    def forward_once(self, xa, xb):
        _, _, h_input, w_input = xa.shape
        assert xa.shape == xb.shape, "The two images are not the same size, please check it."

        fa1, fa2, fa3, fa4 = self.backbone(xa)
        fa1 = self.CAGF1(fa1)
        fa2 = self.CAGF2(fa2)
        fa3 = self.CAGF3(fa3)
        fa4 = self.CAGF4(fa4)
        
        fb1, fb2, fb3, fb4 = self.backbone(xb)
        fb1 = self.CAGF1(fb1)
        fb2 = self.CAGF2(fb2)
        fb3 = self.CAGF3(fb3)
        fb4 = self.CAGF4(fb4)
        
        ms_feats = fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4

        change4, change = self.neck(ms_feats)

        feature_map_4_1 =  self.contrast_loss(change4)
        feature_map_4_1 = self.cv_kan(feature_map_4_1)
    
        out = self.head_forward(ms_feats, change, out_size=(h_input, w_input))

        return feature_map_4_1, out

    def forward_tta(self, xa, xb):
        bs, c, h, w = xa.shape
        mutil_scales = [1.0, 0.834, 0.667, 0.542]

        out1, out2 = 0, 0
        for single_scale in mutil_scales:    # 多尺度测试
            single_scale = (int((h * single_scale) / 32) * 32, int((w * single_scale) / 32) * 32)
            xa_size = F.interpolate(xa, single_scale, mode='bilinear', align_corners=True)
            xb_size = F.interpolate(xb, single_scale, mode='bilinear', align_corners=True)

            out_1 = self.forward_once(xa_size, xb_size)  # 正常forward
            if self.dl:
                out1_1, out1_2 = out_1[0], out_1[1]
            else:
                out1_1 = out_1
                out1 += F.interpolate(out1_1,
                                      size=(h, w), mode='bilinear', align_corners=True)

        return (out1, out2) if self.dl else out1

    def head_forward(self, ms_feats, change, out_size):
        fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4 = ms_feats

        out1 = F.interpolate(self.head1(change), size=out_size, mode='bilinear', align_corners=True)
        out2 = F.interpolate(self.head2(change), size=out_size,
                             mode='bilinear', align_corners=True) if self.dl else None

        if self.training and self.auxiliary_head:
            aux_stage1_out1 = F.interpolate(self.aux_stage1_head1(torch.cat([fa1, fb1], 1)),
                                            size=out_size, mode='bilinear', align_corners=True)
            aux_stage1_out2 = F.interpolate(self.aux_stage1_head2(torch.cat([fa1, fb1], 1)),
                                            size=out_size, mode='bilinear', align_corners=True) if self.dl else None
            aux_stage2_out1 = F.interpolate(self.aux_stage2_head1(torch.cat([fa2, fb2], 1)),
                                            size=out_size, mode='bilinear', align_corners=True)
            aux_stage2_out2 = F.interpolate(self.aux_stage2_head2(torch.cat([fa2, fb2], 1)),
                                            size=out_size, mode='bilinear', align_corners=True) if self.dl else None
            aux_stage3_out1 = F.interpolate(self.aux_stage3_head1(torch.cat([fa3, fb3], 1)),
                                            size=out_size, mode='bilinear', align_corners=True)
            aux_stage3_out2 = F.interpolate(self.aux_stage3_head2(torch.cat([fa3, fb3], 1)),
                                            size=out_size, mode='bilinear', align_corners=True) if self.dl else None
            aux_stage4_out1 = F.interpolate(self.aux_stage4_head1(torch.cat([fa4, fb4], 1)),
                                            size=out_size, mode='bilinear', align_corners=True)
            aux_stage4_out2 = F.interpolate(self.aux_stage4_head2(torch.cat([fa4, fb4], 1)),
                                            size=out_size, mode='bilinear', align_corners=True) if self.dl else None
            return (out1, out2,
                    aux_stage1_out1, aux_stage1_out2, aux_stage2_out1, aux_stage2_out2,
                    aux_stage3_out1, aux_stage3_out2, aux_stage4_out1, aux_stage4_out2) \
                if self.dl else (out1, aux_stage1_out1, aux_stage2_out1, aux_stage3_out1, aux_stage4_out1)
        else:
            return (out1, out2) if self.dl else out1

    def _init_weight(self, pretrain=''):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if pretrain.endswith('.pt'):
            pretrained_dict = torch.load(pretrain)
            if isinstance(pretrained_dict, nn.DataParallel):
                pretrained_dict = pretrained_dict.module
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.state_dict().items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(OrderedDict(model_dict), strict=True)
            print("=> ChangeDetection load {}/{} items from: {}".format(len(pretrained_dict),
                                                                        len(model_dict), pretrain))

    def _model_summary(self, input_size):
        input_sample = torch.randn(1, 3, input_size, input_size)

    def _create_backbone(self, backbone):
        if 'cswin' in backbone:
            if '_t_' in backbone:
                self.backbone = CSWin_64_12211_tiny_224(pretrained=True)
            elif '_s_' in backbone:
                self.backbone = CSWin_64_24322_small_224(pretrained=True)
            elif '_b_' in backbone:
                self.backbone = CSWin_96_24322_base_384(pretrained=True)
            elif '_b448_' in backbone:
                self.backbone = CSWin_96_24322_base_224(pretrained=True)
        else:
            raise Exception('Not Implemented yet: {}'.format(backbone))

    def _create_neck(self, neck):
        if 'fpn' in neck:
            self.neck = FPNNeck(self.inplanes, neck)
    def _select_head(self, head):
        if head == 'fcn':
            return FCNHead(self.inplanes, 2)

    def _create_heads(self, head):
        self.head1 = self._select_head(head)
        self.head2 = self._select_head(head) if self.dl else None

        if self.auxiliary_head:
            self.aux_stage1_head1 = FCNHead(self.inplanes * 2, 2)
            self.aux_stage1_head2 = FCNHead(self.inplanes * 2, 2) if self.dl else None

            self.aux_stage2_head1 = FCNHead(self.inplanes * 4, 2)
            self.aux_stage2_head2 = FCNHead(self.inplanes * 4, 2) if self.dl else None

            self.aux_stage3_head1 = FCNHead(self.inplanes * 8, 2)
            self.aux_stage3_head2 = FCNHead(self.inplanes * 8, 2) if self.dl else None

            self.aux_stage4_head1 = FCNHead(self.inplanes * 16, 2)
            self.aux_stage4_head2 = FCNHead(self.inplanes * 16, 2) if self.dl else None


class EnsembleModel(nn.Module):
    def __init__(self, ckp_paths, device, method="avg2", input_size=512):
        super(EnsembleModel, self).__init__()
        self.method = method
        self.models_list = []
        assert isinstance(ckp_paths, list), "ckp_path must be a list: {}".format(ckp_paths)
        

        print("-"*50+"\n--Ensamble method: {}".format(method))


        for ckp_path in ckp_paths:
            if os.path.isdir(ckp_path):
                weight_files = [f for f in os.listdir(ckp_path) if os.path.isfile(os.path.join(ckp_path, f))]
                print("Files in directory {}: {}".format(ckp_path, weight_files))
                for weight_file in weight_files:
                    full_ckp_path = os.path.join(ckp_path, weight_file)
                    print("--Load model: {}".format(full_ckp_path))
                    model = torch.load(full_ckp_path, map_location=device)
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel) or isinstance(model, nn.DataParallel):
                        model = model.module
                    
                    self.models_list.append(model)
            else:
                print("--Load model: {}".format(ckp_path))
                model = torch.load(ckp_path, map_location=device)
                if isinstance(model, torch.nn.parallel.DistributedDataParallel) or isinstance(model, nn.DataParallel):
                    model = model.module
                self.models_list.append(model)
        self.scale = ScaleInOutput(input_size)

    def eval(self):
        for model in self.models_list:
            model.eval()

    def forward(self, xa, xb, tta=False):
        xa, xb = self.scale.scale_input((xa, xb))
        out1, out2 = 0, 0
        cd_pred1, cd_pred2 = None, None
        for i, model in enumerate(self.models_list):
            _,outs = model(xa, xb, tta)
            if not isinstance(outs, tuple):
                outs = (outs, outs)
            outs = self.scale.scale_output(outs)
            if "avg" in self.method:
                if self.method == "avg2":
                    outs = (F.softmax(outs[0], dim=1), F.softmax(outs[1], dim=1))
                out1 += outs[0]
                out2 += outs[1]
                _, cd_pred1 = torch.max(out1, 1)
                _, cd_pred2 = torch.max(out2, 1)
            elif self.method == "vote":
                _, out1_tmp = torch.max(outs[0], 1)
                _, out2_tmp = torch.max(outs[1], 1)
                out1 += out1_tmp
                out2 += out2_tmp
                cd_pred1 = out1 / i >= 0.5
                cd_pred2 = out2 / i >= 0.5

            elif self.method == "dynamic":
                prob1 = F.softmax(outs[0], dim=1)
                prob2 = F.softmax(outs[1], dim=1)
                entropy1 = -torch.sum(prob1 * torch.log(prob1 + 1e-8), dim=1, keepdim=True)
                entropy2 = -torch.sum(prob2 * torch.log(prob2 + 1e-8), dim=1, keepdim=True)
                weight1 = 1.0 / (entropy1 + 1e-8)
                weight2 = 1.0 / (entropy2 + 1e-8)
                weight_sum = weight1 + weight2
                w1 = weight1 / weight_sum
                w2 = weight2 / weight_sum
                out1 += prob1 * w1
                out2 += prob2 * w2
                _, cd_pred1 = torch.max(out1, 1)
                _, cd_pred2 = torch.max(out2, 1)

        if self.models_list[0].dl:
            return cd_pred1, cd_pred2
        else:
            return cd_pred1                       

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Change Detection train')
    # 配置模型
    parser.add_argument("--backbone", type=str, default="cswin_t_64")
    parser.add_argument("--neck", type=str, default="fpn+drop")
    parser.add_argument("--head", type=str, default="fcn")
    parser.add_argument("--loss", type=str, default="bce+dice")
    parser.add_argument("--pretrain", type=str,default="")
    parser.add_argument("--cuda", default='0', help='whether use CUDA')
    parser.add_argument("--dataset-dir", type=str, default="修改这里")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.00035)
    parser.add_argument("--dual-label", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=True)
    parser.add_argument("--pseudo-label", type=bool, default=False)
    opt = parser.parse_args()
    model = ChangeDetection(opt).cuda()
    in1 = torch.randn(1, 3, 448, 448).cuda()
    in2 = torch.randn(1, 3, 448, 448).cuda()
    edg = model(in1, in2)
    print(edg)