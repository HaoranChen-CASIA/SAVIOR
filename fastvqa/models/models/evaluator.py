import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.nn.functional import adaptive_avg_pool3d
from functools import partial, reduce
from einops import rearrange
import numpy as np
from .swin_backbone import SwinTransformer3D as VideoBackbone
from .swin_backbone import swin_3d_tiny, swin_3d_small
from .conv_backbone import convnext_3d_tiny, convnext_3d_small
from .xclip_backbone import build_x_clip_model
from .swin_backbone import SwinTransformer2D as ImageBackbone
from .swinv2_backbone import SwinTransformerV2
from .swinv1_backbone import SwinTransformer
from .head import VQAHead, IQAHead, VARHead
from .stripformer.networks import get_generator
from .resnet import generate_model
from .core.raft import RAFT

from .resnet2d import resnet50
import fastvqa.models.saverloader as saverloader

from PIL import Image
import cv2
import time
from sklearn.cluster import KMeans

from segment_anything import sam_model_registry  # 用于新的OF支路设计


class BaseEvaluator(nn.Module):
    def __init__(
        self,
        backbone=dict(),
        vqa_head=dict(),
    ):
        super().__init__()
        self.backbone = VideoBackbone(**backbone)
        self.vqa_head = VQAHead(**vqa_head)

    def forward(self, vclip, inference=True, **kwargs):
        if inference:
            self.eval()
            with torch.no_grad():
                feat = self.backbone(vclip)
                score = self.vqa_head(feat)
            self.train()
            return score
        else:
            feat = self.backbone(vclip)
            score = self.vqa_head(feat)
            return score

    def forward_with_attention(self, vclip):
        self.eval()
        with torch.no_grad():
            feat, avg_attns = self.backbone(vclip, require_attn=True)
            score = self.vqa_head(feat)
            return score, avg_attns


class Stablev2Evaluator(nn.Module):  # different part from FastVQA, new code?
    def __init__(
        self,
        backbone_size="divided",
        backbone_preserve_keys = 'fragments,resize',
        multi=False,
        layer=-1,
        backbone=dict(),
        divide_head=False,
        vqa_head=dict(),
    ):
        super().__init__()

        self.multi = multi
        self.layer = layer
        self.blur = 8
        for key, hypers in backbone.items():
            print(backbone_size)
            
            if backbone_size=="divided":
                t_backbone_size = hypers["type"]
            else:
                t_backbone_size = backbone_size
            if t_backbone_size == 'swin_tiny':
                b = swin_3d_tiny(in_chans=2, window_size=(4,4,4), **backbone[key])
            elif t_backbone_size == 'swin_tiny_grpb':
                # to reproduce fast-vqa
                b = VideoBackbone()
            elif t_backbone_size == 'swin_tiny_grpb_m':
                # to reproduce fast-vqa-m
                b = VideoBackbone(window_size=(4,4,4), frag_biases=[0,0,0,0])
            elif t_backbone_size == 'swin_small':
                b = swin_3d_small(**backbone[key])
            elif t_backbone_size == 'conv_tiny':
                b = convnext_3d_tiny(pretrained=True)
            elif t_backbone_size == 'conv_small':
                b = convnext_3d_small(pretrained=True)
            elif t_backbone_size == 'xclip':
                b = build_x_clip_model(**backbone[key])
            elif t_backbone_size == 'swinv2':
                b = SwinTransformerV2(img_size=224, window_size=7, num_classes=128)
            elif t_backbone_size == 'swinv1':
                b = SwinTransformer()
            elif t_backbone_size == 'resnet':
                b = resnet50(pretrained=True)
            else:
                raise NotImplementedError
            print("Setting backbone:", key+"_backbone")
            setattr(self, key+"_backbone", b) 
        
        self.deblur_net = get_generator()
        self.deblur_net.load_state_dict(torch.load('pretrained_weights/Stripformer_realblur_J.pth'))
        self.deblur_net = self.deblur_net.eval()

        self.motion_analyzer = generate_model(18, n_input_channels = 2, n_classes=256)

        self.avg_pool1d = nn.AdaptiveAvgPool1d(1)
        self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.flow_model = RAFT()
        state_dict = torch.load('pretrained_weights/raft-things.pth')
        from collections import OrderedDict
        i_state_dict = OrderedDict()
        for key in state_dict.keys():
            t_key = key.replace("module.", "")
            i_state_dict[t_key] = state_dict[key]
        self.flow_model.load_state_dict(i_state_dict)
        self.flow_model.eval()

        
        self.quality = self.quality_regression(512 + 768 * 32 + 320 * self.blur, 128,1)

    def load_pretrained(self, state_dict):
        t_state_dict = self.resize_backbone.state_dict()
        for key, value in t_state_dict.items():
            if key in state_dict and state_dict[key].shape != value.shape:
                state_dict.pop(key)
        print(self.resize_backbone.load_state_dict(state_dict, strict=False))

    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def get_blur_vec(self, model, frames, num):

        _, d, c, h, w = frames.shape

        with torch.no_grad():
            
            img_tensor = frames[:, 0:d:int(d/num), :, :, :]
            img_tensor = img_tensor.reshape(-1, c, h, w)

            factor = 8
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            img_tensor = F.pad(img_tensor, (0, padw, 0, padh), 'reflect')
            H, W = img_tensor.shape[2], img_tensor.shape[3]

            output = model(img_tensor)
        return output
    
    def forward(self, vclips, inference=True, return_pooled_feats=False, reduce_scores=True, **kwargs):
        if inference:
            self.eval()
            with torch.no_grad():

                scores = []
                feats = {}

                for key in vclips:
                    n, c, d, h, w = vclips[key].shape
                    tmp = rearrange(vclips[key], "n c d h w -> n d c h w")
                    
                    x = vclips[key]
                    x = vclips[key].reshape(-1, c, h, w)
                    optical_flows = []
                
                    with torch.no_grad():
                       for i in range(d):
                           if (i+1 < d):                       
                               flow_up = self.flow_model(vclips[key][:, :, i, :, :], vclips[key][:, :, i+1, :, :])

                           else:
                               flow_up = self.flow_model(vclips[key][:, :, i, :, :], vclips[key][:, :, i, :, :])
                           optical_flows.append(flow_up[0])


                    # TODO: modified in 0830, become same as train process
                    img_f = getattr(self, key.split("_")[0]+"_backbone")(x)  # Watch Out!!
                    # print('\n img_f shape {}'.format(img_f.shape))
                    img_feat = img_f.reshape(n, d*img_f.size(1))

                    # TODO: this code must have something wrong
                    # img_feat = torch.flatten(self.avg_pool3d(img_f), 1)  # strange here, compare with train phase

                    optical_feat = self.motion_analyzer(torch.stack(optical_flows, 2))

                    blur_feats = self.get_blur_vec(self.deblur_net, tmp, self.blur)
                    total_feat = []
                    blur_feats = torch.flatten(self.avg_pool2d(blur_feats), 1)
                    blur_feats = blur_feats.reshape(n, self.blur * blur_feats.size(1))
                    
                    total_feat.append(blur_feats)
                    total_feat.append(img_feat)
                    total_feat.append(optical_feat)

                    total_feat = torch.cat(total_feat, 1)
                    scores += [self.quality(total_feat)]
                    if return_pooled_feats:
                        feats[key] = img_feat
                if reduce_scores:
                    if len(scores) > 1:
                        scores = reduce(lambda x,y:x+y, scores)
                    else:
                        scores = scores[0]
            
                self.train()
                if return_pooled_feats:
                    return scores, feats
                return scores
                
        else:
            self.train()
            scores = []
            feats = {}

            for key in vclips:
                n, c, d, h, w = vclips[key].shape
                tmp = rearrange(vclips[key], "n c d h w -> n d c h w")
                
                x = vclips[key]
                x = vclips[key].reshape(-1, c, h, w)
                optical_flows = []
            
                with torch.no_grad():
                   for i in range(d):
                       if (i+1 < d):                       
                           flow_up = self.flow_model(vclips[key][:, :, i, :, :], vclips[key][:, :, i+1, :, :])
                           
                       else:
                           flow_up = self.flow_model(vclips[key][:, :, i, :, :], vclips[key][:, :, i, :, :])
                       optical_flows.append(flow_up[0])
                    

                img_f = getattr(self, key.split("_")[0]+"_backbone")(x)
                img_feat = img_f.reshape(n, d*img_f.size(1))

                optical_feat = self.motion_analyzer(torch.stack(optical_flows, 2))
            
                blur_feats = self.get_blur_vec(self.deblur_net, tmp, self.blur)
                total_feat = []
                blur_feats = torch.flatten(self.avg_pool2d(blur_feats), 1)
                blur_feats = blur_feats.reshape(n, self.blur * blur_feats.size(1))
                #
                total_feat.append(blur_feats)
                total_feat.append(img_feat)
                total_feat.append(optical_feat)

                total_feat = torch.cat(total_feat, 1)
                scores += [self.quality(total_feat)]
                if return_pooled_feats:
                    feats[key] = img_feat
            if reduce_scores:
                if len(scores) > 1:
                    scores = reduce(lambda x,y:x+y, scores)
                else:
                    scores = scores[0]
            
            if return_pooled_feats:
                return scores, feats
            return scores


    def forward_with_attention(self, vclip):
        self.eval()
        with torch.no_grad():
            feat, avg_attns = self.backbone(vclip, require_attn=True)
            score = self.vqa_head(feat)
            return score, avg_attns


class ssEMEvaluator(nn.Module):  # our model, discard the blur feature, 2024/01/03
    def __init__(
            self,
            backbone_size="divided",
            backbone_preserve_keys='fragments,resize',
            multi=False,
            layer=-1,
            backbone=dict(),
            divide_head=False,
            vqa_head=dict(),
    ):
        super().__init__()

        self.multi = multi
        self.layer = layer
        # self.blur = 8
        for key, hypers in backbone.items():
            print(backbone_size)

            if backbone_size == "divided":
                t_backbone_size = hypers["type"]
            else:
                t_backbone_size = backbone_size
            if t_backbone_size == 'swin_tiny':
                b = swin_3d_tiny(in_chans=2, window_size=(4, 4, 4), **backbone[key])
            elif t_backbone_size == 'swin_tiny_grpb':
                # to reproduce fast-vqa
                b = VideoBackbone()
            elif t_backbone_size == 'swin_tiny_grpb_m':
                # to reproduce fast-vqa-m
                b = VideoBackbone(window_size=(4, 4, 4), frag_biases=[0, 0, 0, 0])
            elif t_backbone_size == 'swin_small':
                b = swin_3d_small(**backbone[key])
            elif t_backbone_size == 'conv_tiny':
                b = convnext_3d_tiny(pretrained=True)
            elif t_backbone_size == 'conv_small':
                b = convnext_3d_small(pretrained=True)
            elif t_backbone_size == 'xclip':
                b = build_x_clip_model(**backbone[key])
            elif t_backbone_size == 'swinv2':
                b = SwinTransformerV2(img_size=224, window_size=7, num_classes=128)
            elif t_backbone_size == 'swinv1':
                b = SwinTransformer()
            elif t_backbone_size == 'resnet':
                b = resnet50(pretrained=True)
            else:
                raise NotImplementedError
            print("Setting backbone:", key + "_backbone")
            setattr(self, key + "_backbone", b)

        # self.deblur_net = get_generator()
        # self.deblur_net.load_state_dict(torch.load('pretrained_weights/Stripformer_realblur_J.pth'))
        # self.deblur_net = self.deblur_net.eval()

        self.motion_analyzer = generate_model(18, n_input_channels=2, n_classes=256)

        self.avg_pool1d = nn.AdaptiveAvgPool1d(1)
        self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.flow_model = RAFT()
        state_dict = torch.load('pretrained_weights/raft-things.pth')
        from collections import OrderedDict
        i_state_dict = OrderedDict()
        for key in state_dict.keys():
            t_key = key.replace("module.", "")
            i_state_dict[t_key] = state_dict[key]
        self.flow_model.load_state_dict(i_state_dict)
        self.flow_model.eval()

        self.quality = self.quality_regression(512 + 768 * 32, 128, 1)  # discarded the blur feature from stable VQA

    def load_pretrained(self, state_dict):
        t_state_dict = self.resize_backbone.state_dict()
        for key, value in t_state_dict.items():
            if key in state_dict and state_dict[key].shape != value.shape:
                state_dict.pop(key)
        print(self.resize_backbone.load_state_dict(state_dict, strict=False))

    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )

        return regression_block

    def forward(self, vclips, inference=True, return_pooled_feats=False, reduce_scores=True, **kwargs):
        if inference:
            self.eval()
            with torch.no_grad():

                scores = []
                feats = {}

                for key in vclips:
                    n, c, d, h, w = vclips[key].shape
                    tmp = rearrange(vclips[key], "n c d h w -> n d c h w")

                    x = vclips[key]
                    x = vclips[key].reshape(-1, c, h, w)
                    optical_flows = []

                    with torch.no_grad():
                        for i in range(d):
                            if (i + 1 < d):
                                flow_up = self.flow_model(vclips[key][:, :, i, :, :], vclips[key][:, :, i + 1, :, :])

                            else:
                                flow_up = self.flow_model(vclips[key][:, :, i, :, :], vclips[key][:, :, i, :, :])
                            optical_flows.append(flow_up[0])

                    # TODO: modified in 0830, become same as train process
                    img_f = getattr(self, key.split("_")[0] + "_backbone")(x)  # Watch Out!!
                    # print('\n img_f shape {}'.format(img_f.shape))
                    img_feat = img_f.reshape(n, d * img_f.size(1))

                    # TODO: this code must have something wrong
                    # img_feat = torch.flatten(self.avg_pool3d(img_f), 1)  # strange here, compare with train phase

                    optical_feat = self.motion_analyzer(torch.stack(optical_flows, 2))

                    # blur_feats = self.get_blur_vec(self.deblur_net, tmp, self.blur)
                    total_feat = []
                    # blur_feats = torch.flatten(self.avg_pool2d(blur_feats), 1)
                    # blur_feats = blur_feats.reshape(n, self.blur * blur_feats.size(1))

                    # total_feat.append(blur_feats)
                    total_feat.append(img_feat)
                    total_feat.append(optical_feat)

                    total_feat = torch.cat(total_feat, 1)
                    scores += [self.quality(total_feat)]
                    if return_pooled_feats:
                        feats[key] = img_feat
                if reduce_scores:
                    if len(scores) > 1:
                        scores = reduce(lambda x, y: x + y, scores)
                    else:
                        scores = scores[0]

                self.train()
                if return_pooled_feats:
                    return scores, feats
                return scores

        else:
            self.train()
            scores = []
            feats = {}

            for key in vclips:
                n, c, d, h, w = vclips[key].shape
                tmp = rearrange(vclips[key], "n c d h w -> n d c h w")

                x = vclips[key]
                x = vclips[key].reshape(-1, c, h, w)
                optical_flows = []

                with torch.no_grad():
                    for i in range(d):
                        if (i + 1 < d):
                            flow_up = self.flow_model(vclips[key][:, :, i, :, :], vclips[key][:, :, i + 1, :, :])

                        else:
                            flow_up = self.flow_model(vclips[key][:, :, i, :, :], vclips[key][:, :, i, :, :])
                        optical_flows.append(flow_up[0])

                img_f = getattr(self, key.split("_")[0] + "_backbone")(x)
                img_feat = img_f.reshape(n, d * img_f.size(1))

                optical_feat = self.motion_analyzer(torch.stack(optical_flows, 2))

                # blur_feats = self.get_blur_vec(self.deblur_net, tmp, self.blur)
                total_feat = []
                # blur_feats = torch.flatten(self.avg_pool2d(blur_feats), 1)
                # blur_feats = blur_feats.reshape(n, self.blur * blur_feats.size(1))
                #
                # total_feat.append(blur_feats)
                total_feat.append(img_feat)
                total_feat.append(optical_feat)

                total_feat = torch.cat(total_feat, 1)
                scores += [self.quality(total_feat)]
                if return_pooled_feats:
                    feats[key] = img_feat
            if reduce_scores:
                if len(scores) > 1:
                    scores = reduce(lambda x, y: x + y, scores)
                else:
                    scores = scores[0]

            if return_pooled_feats:
                return scores, feats
            return scores

    def forward_with_attention(self, vclip):
        self.eval()
        with torch.no_grad():
            feat, avg_attns = self.backbone(vclip, require_attn=True)
            score = self.vqa_head(feat)
            return score, avg_attns


class ssEMevaluator_ablation(nn.Module):  # different part from FastVQA, new code?
    def __init__(
            self,
            feat_type='sf+of+bf',
            backbone_size="divided",
            backbone_preserve_keys='fragments,resize',
            multi=False,
            layer=-1,
            backbone=dict(),
            divide_head=False,
            vqa_head=dict(),
    ):
        super().__init__()

        self.multi = multi
        self.layer = layer
        self.blur = 8
        self.feat_type = feat_type
        print('Feature Selected: {}'.format(self.feat_type))

        # phase 1: set semantic feature extractor backbone; can we replace swin-v1 or input with mask?
        for key, hypers in backbone.items():
            print(backbone_size)

            if backbone_size == "divided":
                t_backbone_size = hypers["type"]
            else:
                t_backbone_size = backbone_size
            if t_backbone_size == 'swin_tiny':
                b = swin_3d_tiny(in_chans=2, window_size=(4, 4, 4), **backbone[key])
            elif t_backbone_size == 'swin_tiny_grpb':
                # to reproduce fast-vqa
                b = VideoBackbone()
            elif t_backbone_size == 'swin_tiny_grpb_m':
                # to reproduce fast-vqa-m
                b = VideoBackbone(window_size=(4, 4, 4), frag_biases=[0, 0, 0, 0])
            elif t_backbone_size == 'swin_small':
                b = swin_3d_small(**backbone[key])
            elif t_backbone_size == 'conv_tiny':
                b = convnext_3d_tiny(pretrained=True)
            elif t_backbone_size == 'conv_small':
                b = convnext_3d_small(pretrained=True)
            elif t_backbone_size == 'xclip':
                b = build_x_clip_model(**backbone[key])
            elif t_backbone_size == 'swinv2':
                b = SwinTransformerV2(img_size=224, window_size=7, num_classes=128)
            elif t_backbone_size == 'swinv1':
                b = SwinTransformer()
            elif t_backbone_size == 'resnet':
                b = resnet50(pretrained=True)
            else:
                raise NotImplementedError
            print("Setting backbone:", key + "_backbone")
            setattr(self, key + "_backbone", b)

        # phase 2: set blur feature extractor
        self.deblur_net = get_generator()
        self.deblur_net.load_state_dict(torch.load('pretrained_weights/Stripformer_realblur_J.pth'))
        self.deblur_net = self.deblur_net.eval()

        # phase 3.1: set motion feature extractor
        self.motion_analyzer = generate_model(18, n_input_channels=2, n_classes=256)

        # todo: Strange:这部分很奇怪，在stable VQA原代码中似乎也没用上，什么情况？
        self.avg_pool1d = nn.AdaptiveAvgPool1d(1)
        self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))

        # phase 3.2: set optical flow extractor; can we replace RAFT with something else?
        self.flow_model = RAFT()
        state_dict = torch.load('pretrained_weights/raft-things.pth')
        from collections import OrderedDict
        i_state_dict = OrderedDict()
        for key in state_dict.keys():
            t_key = key.replace("module.", "")
            i_state_dict[t_key] = state_dict[key]
        self.flow_model.load_state_dict(i_state_dict)
        self.flow_model.eval()

        # set MLP according to the selected feature
        self.quality = self.quality_regression(512 + 768 * 32 + 320 * self.blur, 128, 1)
        if self.feat_type == 'sf':  # only semantic feat
            self.quality = self.quality_regression(768 * 32, 128, 1)
        elif self.feat_type == 'of':  # only optical-flow feat
            self.quality = self.quality_regression(512, 128, 1)
        elif self.feat_type == 'bf':  # only motion blur feat
            self.quality = self.quality_regression(320 * self.blur, 128, 1)
        elif self.feat_type == 'sf+of':  # sf + of
            self.quality = self.quality_regression(512 + 768 * 32, 128, 1)
        elif self.feat_type == 'sf+bf':  # sf + bf
            self.quality = self.quality_regression(768 * 32 + 320 * self.blur, 128, 1)
        elif self.feat_type == 'of+bf':  # of + bf
            self.quality = self.quality_regression(512 + 320 * self.blur, 128, 1)

    def load_pretrained(self, state_dict):
        t_state_dict = self.resize_backbone.state_dict()
        for key, value in t_state_dict.items():
            if key in state_dict and state_dict[key].shape != value.shape:
                state_dict.pop(key)
        print(self.resize_backbone.load_state_dict(state_dict, strict=False))

    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )

        return regression_block

    def get_blur_vec(self, model, frames, num):

        _, d, c, h, w = frames.shape

        with torch.no_grad():
            img_tensor = frames[:, 0:d:int(d / num), :, :, :]
            img_tensor = img_tensor.reshape(-1, c, h, w)

            factor = 8
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            img_tensor = F.pad(img_tensor, (0, padw, 0, padh), 'reflect')
            H, W = img_tensor.shape[2], img_tensor.shape[3]

            output = model(img_tensor)
        return output

    def forward(self, vclips, inference=True, return_pooled_feats=False, reduce_scores=True, return_Feat=False, **kwargs):
        if inference:
            self.eval()
            with torch.no_grad():

                scores = []
                feats = {}

                for key in vclips:
                    n, c, d, h, w = vclips[key].shape
                    tmp = rearrange(vclips[key], "n c d h w -> n d c h w")

                    x = vclips[key]
                    x = vclips[key].reshape(-1, c, h, w)
                    optical_flows = []

                    if 'of' in self.feat_type:
                        with torch.no_grad():
                            for i in range(d):
                                if (i + 1 < d):
                                    # flow_model接受的输入是2个（B,3,224,224）的输入
                                    flow_up = self.flow_model(vclips[key][:, :, i, :, :],
                                                              vclips[key][:, :, i + 1, :, :])

                                else:
                                    flow_up = self.flow_model(vclips[key][:, :, i, :, :], vclips[key][:, :, i, :, :])
                                optical_flows.append(flow_up[0])

                    # TODO: select feature type according to setting
                    total_feat = []
                    if 'sf' in self.feat_type:
                        img_f = getattr(self, key.split("_")[0] + "_backbone")(x)  # Watch Out!!
                        img_feat = img_f.reshape(n, d * img_f.size(1))
                        total_feat.append(img_feat)

                    if 'of' in self.feat_type:
                        optical_feat = self.motion_analyzer(torch.stack(optical_flows, 2))
                        total_feat.append(optical_feat)

                    if 'bf' in self.feat_type:
                        blur_feats = self.get_blur_vec(self.deblur_net, tmp, self.blur)
                        blur_feats = torch.flatten(self.avg_pool2d(blur_feats), 1)
                        blur_feats = blur_feats.reshape(n, self.blur * blur_feats.size(1))
                        total_feat.append(blur_feats)

                    all_feats = total_feat
                    total_feat = torch.cat(total_feat, 1)
                    scores += [self.quality(total_feat)]
                    if return_pooled_feats:
                        feats[key] = img_feat
                if reduce_scores:
                    if len(scores) > 1:
                        scores = reduce(lambda x, y: x + y, scores)
                    else:
                        scores = scores[0]

                self.train()
                if return_pooled_feats:
                    return scores, feats
                if return_Feat:  # TODO: 20250604追加，输出分数与特征
                    return scores, all_feats
                else:
                    return scores

        else:
            self.train()
            scores = []
            feats = {}

            for key in vclips:
                n, c, d, h, w = vclips[key].shape
                tmp = rearrange(vclips[key], "n c d h w -> n d c h w")

                x = vclips[key]
                x = vclips[key].reshape(-1, c, h, w)
                optical_flows = []

                if 'of' in self.feat_type:
                    with torch.no_grad():
                        for i in range(d):
                            if (i + 1 < d):
                                flow_up = self.flow_model(vclips[key][:, :, i, :, :], vclips[key][:, :, i + 1, :, :])

                            else:
                                flow_up = self.flow_model(vclips[key][:, :, i, :, :], vclips[key][:, :, i, :, :])
                            optical_flows.append(flow_up[0])

                total_feat = []
                if 'sf' in self.feat_type:
                    img_f = getattr(self, key.split("_")[0] + "_backbone")(x)
                    img_feat = img_f.reshape(n, d * img_f.size(1))
                    total_feat.append(img_feat)

                if 'of' in self.feat_type:
                    optical_feat = self.motion_analyzer(torch.stack(optical_flows, 2))
                    total_feat.append(optical_feat)

                if 'bf' in self.feat_type:
                    blur_feats = self.get_blur_vec(self.deblur_net, tmp, self.blur)
                    blur_feats = torch.flatten(self.avg_pool2d(blur_feats), 1)
                    blur_feats = blur_feats.reshape(n, self.blur * blur_feats.size(1))
                    total_feat.append(blur_feats)

                total_feat = torch.cat(total_feat, 1)
                scores += [self.quality(total_feat)]
                if return_pooled_feats:
                    feats[key] = img_feat
            if reduce_scores:
                if len(scores) > 1:
                    scores = reduce(lambda x, y: x + y, scores)
                else:
                    scores = scores[0]

            if return_pooled_feats:
                return scores, feats
            return scores

    def forward_with_attention(self, vclip):
        self.eval()
        with torch.no_grad():
            feat, avg_attns = self.backbone(vclip, require_attn=True)
            score = self.vqa_head(feat)
            return score, avg_attns


class SAM_OFbranch(nn.Module):
    def __init__(
            self,
            feat_type='SAM+of',
            backbone_size="divided",
            backbone_preserve_keys='fragments,resize',
            multi=False,
            layer=-1,
            backbone=dict(),
            divide_head=False,
            vqa_head=dict(),
    ):
        super().__init__()

        self.feat_type = feat_type
        print('Feature Selected: {}'.format(self.feat_type))

        # # phase 3.1: set motion feature extractor
        # self.motion_analyzer = generate_model(18, n_input_channels=2, n_classes=256)
        # self.sam_cossim_analyzer = generate_model(18, n_input_channels=1, n_classes=256)

        # 20250624 new test, 将所有信息叠加到一个3D输入交由统一的3D resnet处理
        self.motion_analyzer_with_obj_info = generate_model(18, n_input_channels=3, n_classes=256)

        # phase 3.2: set optical flow extractor; can we replace RAFT with something else?
        self.flow_model = RAFT()
        state_dict = torch.load('pretrained_weights/raft-things.pth')
        from collections import OrderedDict
        i_state_dict = OrderedDict()
        for key in state_dict.keys():
            t_key = key.replace("module.", "")
            i_state_dict[t_key] = state_dict[key]
        self.flow_model.load_state_dict(i_state_dict)
        self.flow_model.eval()

        # TODO: phase 3.3: 需要配置SAM环境，实现实时的SAM embedding提取
        sam_model_type = 'vit_h'
        sam_checkpoint = '/opt/data/3_107/chenhr/python_codes/segment-anything/pretrained_weights/sam_vit_h_4b8939.pth'
        # sam_checkpoint = '/mnt/Ext001/chenhr/SAM_related/SAM_local/pretrained_weights/sam_vit_h_4b8939.pth'

        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)

        # set MLP according to the selected feature
        # self.quality = self.quality_regression(512+512, 128, 1)
        self.quality = self.quality_regression(512, 128, 1)

    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    @staticmethod
    def upscale_tensor(input_tensor, target_size=(1024, 1024)):
        output_tensor = F.interpolate(
            input_tensor,
            size=target_size,  # 目标尺寸
            mode='bilinear',  # 插值方式
            align_corners=False  # 对齐方式（与OpenCV兼容设为False）
        )
        return output_tensor

    @staticmethod
    def spatial_cosine_similarity(feat1, feat2):
        """
        计算两个特征图的空间点对点余弦相似度

        参数:
            feat1: 形状 (1, 256, 64, 64) 的特征图1
            feat2: 形状 (1, 256, 64, 64) 的特征图2

        返回:
            相似度图 (1, 1, 64, 64)
        """
        # 归一化特征向量（沿通道维度）
        feat1_norm = F.normalize(feat1, p=2, dim=1)  # (1, 256, 64, 64)
        feat2_norm = F.normalize(feat2, p=2, dim=1)  # (1, 256, 64, 64)

        # 点积计算相似度
        similarity_map = (feat1_norm * feat2_norm).sum(dim=1, keepdim=True)  # (1, 1, 64, 64)

        return similarity_map

    def sam_cosine_similarity(self, v_ref, v_mov):
        """
            输入的v_ref和v_mov为两个tensor，shape为(batch_size,3,224,224)
        """
        # ts = time.perf_counter()
        sam_in_ref = self.upscale_tensor(v_ref)
        sam_in_mov = self.upscale_tensor(v_mov)
        # te = time.perf_counter()
        # print('tensor upscaling time: {:.3f}s'.format(te - ts))

        # ts = time.perf_counter()
        embed_ref = self.sam.image_encoder(sam_in_ref)
        embed_mov = self.sam.image_encoder(sam_in_mov)
        # te = time.perf_counter()
        # print('SAM embedding time: {:.3f}s'.format(te - ts))  # TODO: 250626已确认这部分为核心时间消耗区，约4.5s耗费于此

        # ts = time.perf_counter()
        cos_sim_map = self.spatial_cosine_similarity(embed_ref, embed_mov)
        # te = time.perf_counter()
        # print('SAM cos-sim map computation time: {:.3f}s'.format(te - ts))

        return self.upscale_tensor(cos_sim_map, (224, 224))

    def forward(self, vclips, inference=True, return_pooled_feats=False, reduce_scores=True, return_Feat=False, **kwargs):
        # TODO: 需要重新确认这部分代码的运行时间，目前处理时间大概150+s/it，远远低于之前设计的版本
        if inference:
            self.eval()
            with torch.no_grad():

                scores = []
                feats = {}

                for key in vclips:
                    n, c, d, h, w = vclips[key].shape
                    tmp = rearrange(vclips[key], "n c d h w -> n d c h w")

                    x = vclips[key]
                    x = vclips[key].reshape(-1, c, h, w)
                    # optical_flows = []
                    # sam_cossim_maps = []
                    of_with_cmaps = []

                    # ts = time.perf_counter()
                    if 'of' in self.feat_type:
                        with torch.no_grad():
                            for i in range(d):
                                if (i + 1 < d):
                                    flow_up = self.flow_model(vclips[key][:, :, i, :, :],
                                                              vclips[key][:, :, i + 1, :, :])
                                    # ts = time.perf_counter()
                                    cossim_map = self.sam_cosine_similarity(vclips[key][:, :, i, :, :],
                                                                            vclips[key][:, :, i + 1, :, :])
                                    # te = time.perf_counter()
                                    # print('SAM cos-map computation time: {:.3f}s'.format(te - ts))

                                else:
                                    flow_up = self.flow_model(vclips[key][:, :, i, :, :], vclips[key][:, :, i, :, :])
                                    cossim_map = self.sam_cosine_similarity(vclips[key][:, :, i, :, :],
                                                                            vclips[key][:, :, i, :, :])
                                # optical_flows.append(flow_up[0])
                                # sam_cossim_maps.append(cossim_map)
                                of_with_cmaps.append(torch.cat([flow_up[0], cossim_map], dim=1))
                    # te = time.perf_counter()
                    # print('Optical flow & corresponding SAM cosine similarity '
                    #       'map extraction time: {:.3f}s'.format(te - ts))

                    # TODO: select feature type according to setting
                    total_feat = []

                    # if 'of' in self.feat_type:
                    #     optical_feat = self.motion_analyzer(torch.stack(optical_flows, 2))
                    #     total_feat.append(optical_feat)
                    #     sam_cossim_feat = self.sam_cossim_analyzer(torch.stack(sam_cossim_maps, 2))
                    #     total_feat.append(sam_cossim_feat)

                    # ts = time.perf_counter()
                    of_with_object_info = self.motion_analyzer_with_obj_info(torch.stack(of_with_cmaps, 2))
                    # te = time.perf_counter()
                    # print('Combined feature extraction time: {:.3f}s'.format(te - ts))
                    total_feat.append(of_with_object_info)

                    all_feats = total_feat
                    total_feat = torch.cat(total_feat, 1)
                    scores += [self.quality(total_feat)]

                if reduce_scores:
                    if len(scores) > 1:
                        scores = reduce(lambda x, y: x + y, scores)
                    else:
                        scores = scores[0]

                self.train()
                if return_Feat:  # TODO: 输出分数与特征
                    return scores, all_feats
                else:
                    return scores

        else:
            self.train()
            scores = []
            feats = {}

            for key in vclips:
                n, c, d, h, w = vclips[key].shape
                tmp = rearrange(vclips[key], "n c d h w -> n d c h w")

                x = vclips[key]
                x = vclips[key].reshape(-1, c, h, w)
                # optical_flows = []
                # sam_cossim_maps = []
                of_with_cmaps = []

                if 'of' in self.feat_type:
                    with torch.no_grad():
                        for i in range(d):
                            if (i + 1 < d):
                                flow_up = self.flow_model(vclips[key][:, :, i, :, :],
                                                          vclips[key][:, :, i + 1, :, :])
                                cossim_map = self.sam_cosine_similarity(vclips[key][:, :, i, :, :],
                                                                        vclips[key][:, :, i + 1, :, :])

                            else:
                                flow_up = self.flow_model(vclips[key][:, :, i, :, :], vclips[key][:, :, i, :, :])
                                cossim_map = self.sam_cosine_similarity(vclips[key][:, :, i, :, :],
                                                                        vclips[key][:, :, i, :, :])
                            # optical_flows.append(flow_up[0])
                            # sam_cossim_maps.append(cossim_map)
                            of_with_cmaps.append(torch.cat([flow_up[0], cossim_map], dim=1))

                total_feat = []

                # if 'of' in self.feat_type:
                #     optical_feat = self.motion_analyzer(torch.stack(optical_flows, 2))
                #     total_feat.append(optical_feat)
                #     sam_cossim_feat = self.sam_cossim_analyzer(torch.stack(sam_cossim_maps, 2))
                #     total_feat.append(sam_cossim_feat)

                of_with_object_info = self.motion_analyzer_with_obj_info(torch.stack(of_with_cmaps, 2))
                # print(of_with_object_info.shape)
                total_feat.append(of_with_object_info)

                total_feat = torch.cat(total_feat, 1)
                scores += [self.quality(total_feat)]

            if reduce_scores:
                if len(scores) > 1:
                    scores = reduce(lambda x, y: x + y, scores)
                else:
                    scores = scores[0]

            if return_Feat:
                return scores, total_feat
            return scores

from skimage.transform import resize
class SAM_OF_Divide_branch(nn.Module):
    def __init__(
            self,
            feat_type='SAM+of',
            backbone_size="divided",
            backbone_preserve_keys='fragments,resize',
            multi=False,
            layer=-1,
            backbone=dict(),
            divide_head=False,
            vqa_head=dict(),
    ):
        super().__init__()

        self.feat_type = feat_type
        print('Feature Selected: {}'.format(self.feat_type))

        # 20250628 new test, 根据3区域划分，每个区域的光流分别输入到一个3d resnet
        # 按mask区域大小排布，暂时先划分为3区域测试一下
        self.MA_a0 = generate_model(18, n_input_channels=2, n_classes=256)
        self.MA_a1 = generate_model(18, n_input_channels=2, n_classes=256)
        self.MA_a2 = generate_model(18, n_input_channels=2, n_classes=256)

        # phase 3.2: set optical flow extractor; can we replace RAFT with something else?
        self.flow_model = RAFT()
        state_dict = torch.load('pretrained_weights/raft-things.pth')
        from collections import OrderedDict
        i_state_dict = OrderedDict()
        for key in state_dict.keys():
            t_key = key.replace("module.", "")
            i_state_dict[t_key] = state_dict[key]
        self.flow_model.load_state_dict(i_state_dict)
        self.flow_model.eval()

        # TODO: phase 3.3: 需要配置SAM环境，实现实时的SAM embedding提取
        sam_model_type = 'vit_h'
        # sam_checkpoint = '/opt/data/3_107/chenhr/python_codes/segment-anything/pretrained_weights/sam_vit_h_4b8939.pth'
        sam_checkpoint = '/mnt/Ext001/chenhr/SAM_related/SAM_local/pretrained_weights/sam_vit_h_4b8939.pth'

        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)

        # set MLP according to the selected feature
        self.quality = self.quality_regression(512+512+512, 128, 1)

    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    @staticmethod
    def upscale_tensor(input_tensor, target_size=(1024, 1024)):
        output_tensor = F.interpolate(
            input_tensor,
            size=target_size,  # 目标尺寸
            mode='bilinear',  # 插值方式
            align_corners=False  # 对齐方式（与OpenCV兼容设为False）
        )
        return output_tensor

    @staticmethod
    def inter_layer_cluster_v1(feat_map1, feat_map2, n_clusters=3):
        """
        输入两个特征图Tensor，计算特征相似性并进行聚类

        参数:
            feat_map1: 形状 (C, H, W) 的特征图1 (PyTorch Tensor)
            feat_map2: 形状 (C, H, W) 的特征图2 (PyTorch Tensor)
            n_clusters: 聚类数量
        """
        # TODO：这部分代码在3.91不知为何运行极其缓慢，需要调整，最好能改成纯gpu运算
        # 确保输入是Tensor且形状一致
        assert isinstance(feat_map1, torch.Tensor) and isinstance(feat_map2, torch.Tensor), "输入必须是PyTorch Tensor"
        assert feat_map1.shape == feat_map2.shape, "特征图形状必须相同"

        C, H, W = feat_map1.shape

        # 步骤1：展平特征图 (C, H*W)
        flat1 = feat_map1.reshape(C, -1)  # (C, H*W)
        flat2 = feat_map2.reshape(C, -1)  # (C, H*W)

        # 步骤2：计算特征相似性矩阵 (H*W, H*W)
        # 使用PyTorch的余弦相似度计算
        # flat1_norm = F.normalize(flat1, p=2, dim=0)  # 沿通道维度归一化
        # flat2_norm = F.normalize(flat2, p=2, dim=0)
        # similarity = torch.mm(flat1_norm.T, flat2_norm)  # (H*W, H*W)

        # 步骤3：构建联合特征表示 (H*W, 2C)
        joint_features = torch.cat([flat1.T, flat2.T], dim=1)  # (H*W, 2C)

        # 转换为numpy供sklearn使用（如需要GPU加速可跳过此步）
        joint_features_np = joint_features.cpu().numpy()

        # 步骤4：归一化处理
        from sklearn.preprocessing import Normalizer
        normalized_features = Normalizer(norm='l2').fit_transform(joint_features_np)

        # 步骤5：K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        labels = kmeans.fit_predict(normalized_features)  # (H*W,)

        # 步骤6：将聚类结果还原为空间布局
        cluster_map = labels.reshape(H, W)  # 形状 (H, W)

        return cluster_map

    @staticmethod
    def inter_layer_cluster_gpu(feat_map1, feat_map2, n_clusters=3, max_iters=100, tol=1e-4):
        """
        GPU加速的层间聚类实现 (完全PyTorch实现)

        参数:
            feat_map1: (C, H, W) 特征图1
            feat_map2: (C, H, W) 特征图2
            n_clusters: 聚类数量
            max_iters: 最大迭代次数
            tol: 收敛阈值
        返回:
            cluster_map: (H, W) 的聚类结果
        """
        assert feat_map1.shape == feat_map2.shape, "特征图形状必须相同"
        C, H, W = feat_map1.shape

        # 步骤1：展平并转置为 (H*W, 2C) 的联合特征
        joint_features = torch.cat([
            feat_map1.reshape(C, -1).T,  # (H*W, C)
            feat_map2.reshape(C, -1).T  # (H*W, C)
        ], dim=1)  # (H*W, 2C)

        # 步骤2：L2归一化 (替代sklearn Normalizer)
        normalized_features = F.normalize(joint_features, p=2, dim=1)  # (H*W, 2C)

        # 步骤3：GPU加速的K-means (PyTorch实现)
        def kmeans_gpu(X, k, max_iters, tol):
            # 随机初始化聚类中心
            centroids = X[torch.randperm(X.shape[0])[:k]].clone()  # (k, 2C)

            for _ in range(max_iters):
                # 计算距离矩阵 (广播优化)
                distances = torch.cdist(X, centroids, p=2)  # (H*W, k)

                # 分配标签
                labels = torch.argmin(distances, dim=1)  # (H*W,)

                # 计算新中心点
                new_centroids = torch.stack([
                    X[labels == i].mean(dim=0)
                    for i in range(k)
                ])  # (k, 2C)

                # 检查收敛
                if torch.norm(centroids - new_centroids) < tol:
                    break
                centroids = new_centroids

            return labels

        # 执行聚类
        labels = kmeans_gpu(normalized_features, n_clusters, max_iters, tol)

        def relabel_by_area_fast(labels):
            counts = torch.bincount(labels)
            sorted_labels = torch.argsort(counts, descending=True)
            return sorted_labels.argsort()[labels]

        sorted_labels = relabel_by_area_fast(labels)

        # 步骤4：还原空间布局
        return sorted_labels.reshape(H, W)  # (H, W)

    def sam_area_divide(self, v_ref, v_mov):
        """
            输入的v_ref和v_mov为两个tensor，shape为(batch_size,3,224,224)
        """
        batch_size = v_ref.shape[0]
        sam_in_ref = self.upscale_tensor(v_ref)
        sam_in_mov = self.upscale_tensor(v_mov)

        # self.sam.to(device="cuda")
        # ts = time.perf_counter()
        embed_ref = self.sam.image_encoder(sam_in_ref)
        embed_mov = self.sam.image_encoder(sam_in_mov)
        # te = time.perf_counter()
        # print('SAM EMBEDDING TIME {:.2f}'.format(te - ts))  # TODO: Check how much time spend here!

        batch_cluster_map = []
        for b in range(batch_size):
            # cluster_map = self.inter_layer_cluster_v1(embed_ref[b], embed_mov[b])  # (64,64) ndarray
            cluster_map = self.inter_layer_cluster_gpu(embed_ref[b], embed_mov[b])  # tensor
            # print('Unique masks {}'.format(np.unique(cluster_map)))
            batch_cluster_map.append(cluster_map)

        # resized_masks = self.process_mask_arrays(batch_cluster_map)
        resized_masks = self.process_mask_tensors(torch.stack(batch_cluster_map, dim=0))

        return resized_masks

    @staticmethod
    def process_mask_arrays(mask_list, target_size=(224, 224)):
        """
        处理掩码列表到目标尺寸

        参数:
            mask_list: 包含N个(H,W)数组的列表，值为离散整数
            target_size: 目标尺寸(height, width)

        返回:
            形状(N, target_height, target_width)的ndarray
        """
        # 合并数组
        merged = np.stack(mask_list, axis=0)

        # 调整尺寸
        resized = np.zeros((len(mask_list), *target_size), dtype=merged.dtype)
        for i in range(len(mask_list)):
            resized[i] = resize(
                merged[i],
                target_size,
                order=0,  # 最近邻插值避免产生额外label
                preserve_range=True
            ).astype(merged.dtype)

        # print('Resized mask unique elements {}'.format(np.unique(resized)))
        # print('Resized mask shape {}'.format(resized.shape))

        return resized

    @staticmethod
    def process_mask_tensors(mask_tensor, target_size=(224, 224)):
        """
        处理掩码Tensor到目标尺寸 (PyTorch GPU版本)

        参数:
            mask_tensor: 形状为 (N, H, W) 的PyTorch Tensor，值为离散整数
            target_size: 目标尺寸 (height, width)

        返回:
            形状 (N, target_height, target_width) 的Tensor
        """
        # 输入验证
        assert isinstance(mask_tensor, torch.Tensor), "输入必须是PyTorch Tensor"
        assert mask_tensor.dtype in [torch.int32, torch.int64, torch.uint8], "掩码应为整数类型"

        # N, H, W = mask_tensor.shape
        # target_h, target_w = target_size

        # 转换为浮点型用于插值 (保持设备一致)
        mask_float = mask_tensor.float()

        # 调整尺寸 (使用最近邻插值)
        resized = F.interpolate(
            mask_float.unsqueeze(1),  # 添加通道维 (N, 1, H, W)
            size=target_size,
            mode='nearest'
        ).squeeze(1)  # 移除通道维 (N, H, W)

        # 转换回原始数据类型
        return resized.to(mask_tensor.dtype)

    @staticmethod
    def split_tensor_vectorized(tensor, mask_tensor):
        """向量化实现（更高效）"""
        # mask_tensor = torch.from_numpy(mask) if isinstance(mask, np.ndarray) else mask
        mask_expanded = mask_tensor.expand_as(tensor)  # (4,2,32,224,224)

        # 一次性生成所有类别的掩码
        unique_labels, counts = torch.unique(mask_tensor, return_counts=True)
        class_masks = [(mask_expanded == i).float() for i in range(len(counts))]

        return [tensor * mask for mask in class_masks]

    def forward(self, vclips, inference=True, reduce_scores=True, return_Feat=False):
        if inference:
            self.eval()
            with torch.no_grad():
                scores = []

                for key in vclips:
                    n, c, d, h, w = vclips[key].shape
                    optical_flows = []
                    sam_area_masks = []

                    with torch.no_grad():
                        for i in range(d):
                            if (i + 1 < d):
                                flow_up = self.flow_model(vclips[key][:, :, i, :, :],
                                                          vclips[key][:, :, i + 1, :, :])
                                cluster_maps = self.sam_area_divide(vclips[key][:, :, i, :, :],
                                                                    vclips[key][:, :, i + 1, :, :])

                            else:
                                flow_up = self.flow_model(vclips[key][:, :, i, :, :], vclips[key][:, :, i, :, :])
                                cluster_maps = self.sam_area_divide(vclips[key][:, :, i, :, :],
                                                                    vclips[key][:, :, i, :, :])
                            optical_flows.append(flow_up[0])
                            sam_area_masks.append(cluster_maps.unsqueeze(dim=1))  #

                    total_feat = []

                    full_optical_flows = torch.stack(optical_flows, 2)
                    full_cluster_maps = torch.stack(sam_area_masks, 2)
                    split_optical_flows = self.split_tensor_vectorized(full_optical_flows, full_cluster_maps)

                    # # Largest Area
                    optical_feat = self.MA_a0(split_optical_flows[0])
                    total_feat.append(optical_feat)
                    #
                    # Middle Area
                    optical_feat = self.MA_a1(split_optical_flows[1])
                    total_feat.append(optical_feat)
                    #
                    # Smallest Area
                    optical_feat = self.MA_a2(split_optical_flows[2])
                    total_feat.append(optical_feat)

                    all_feats = total_feat
                    total_feat = torch.cat(total_feat, 1)
                    scores += [self.quality(total_feat)]

                if reduce_scores:
                    if len(scores) > 1:
                        scores = reduce(lambda x, y: x + y, scores)
                    else:
                        scores = scores[0]

                self.train()
                if return_Feat:  # TODO: 输出分数与特征
                    return scores, all_feats
                else:
                    return scores

        else:
            self.train()
            scores = []

            for key in vclips:
                n, c, d, h, w = vclips[key].shape
                optical_flows = []
                sam_area_masks = []

                with torch.no_grad():
                    for i in range(d):
                        if (i + 1 < d):
                            flow_up = self.flow_model(vclips[key][:, :, i, :, :],
                                                      vclips[key][:, :, i + 1, :, :])
                            cluster_maps = self.sam_area_divide(vclips[key][:, :, i, :, :],
                                                                vclips[key][:, :, i + 1, :, :])

                        else:
                            flow_up = self.flow_model(vclips[key][:, :, i, :, :], vclips[key][:, :, i, :, :])
                            cluster_maps = self.sam_area_divide(vclips[key][:, :, i, :, :],
                                                                vclips[key][:, :, i, :, :])
                        optical_flows.append(flow_up[0])
                        sam_area_masks.append(cluster_maps.unsqueeze(dim=1))  #

                total_feat = []

                full_optical_flows = torch.stack(optical_flows, 2)
                full_cluster_maps = torch.stack(sam_area_masks, 2)
                split_optical_flows = self.split_tensor_vectorized(full_optical_flows, full_cluster_maps)

                # # Largest Area
                optical_feat = self.MA_a0(split_optical_flows[0])
                total_feat.append(optical_feat)
                #
                # Middle Area
                optical_feat = self.MA_a1(split_optical_flows[1])
                total_feat.append(optical_feat)
                #
                # Smallest Area
                optical_feat = self.MA_a2(split_optical_flows[2])
                total_feat.append(optical_feat)

                all_feats = total_feat
                total_feat = torch.cat(total_feat, 1)
                scores += [self.quality(total_feat)]

            if reduce_scores:
                if len(scores) > 1:
                    scores = reduce(lambda x, y: x + y, scores)
                else:
                    scores = scores[0]

            if return_Feat:
                return scores, all_feats
            return scores

