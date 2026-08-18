##############################################################################################
# Note that our implementation is based on IMDL-Benco[1],
# and the complete source code will be made publicly available upon acceptance of this paper.
# [1] Ma, Xiaochen and Zhu, Xuekang and Su, Lei and Du, Bo and others. IMDL-Benco: A comprehensive benchmark and codebase for image manipulation detection and localization. In NeurIPS, 2025.
##############################################################################################

import torch
import torch.nn as nn
from yacs.config import CfgNode as CN
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from .seg_hrnet_config import get_hrnet_cfg
from .seg_hrnet import get_seg_model
from .Prfm import NLCDetection
from .Pre_edge import EGRM
from .detection_head import DetectionHead
from IMDLBenCo.registry import MODELS

def dice_loss(pred, target, smooth=1.0):
    
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    denominator = (target_flat * target_flat).sum() + (pred_flat * pred_flat).sum()
    
    dice_coeff = (2. * intersection + smooth) / (denominator + smooth)
    
    return 1 - dice_coeff

@MODELS.register_module()
class LapRLNet(nn.Module):
    def __init__(self,
                 input_size: int = 256,
                 pretrain_path: str = None
                 ):
        super(LapRLNet , self).__init__()
        self.FENet = get_seg_model(get_hrnet_cfg(pretrain_path))
        self.args = {'crop_size': [input_size, input_size]}
        self.SegNet = NLCDetection(self.args)
        self.ClsNet = DetectionHead(self.args)
        self.EdgeModule = EGRM()

    def generate_4mask(self, mask):
        mask2 = TF.resize(mask, (mask.shape[2] // 2, mask.shape[3] // 2), antialias=True)
        mask3 = TF.resize(mask, (mask.shape[2] // 4, mask.shape[3] // 4), antialias=True)
        mask4 = TF.resize(mask, (mask.shape[2] // 8, mask.shape[3] // 8), antialias=True)

        mask = (mask > 0.5).float()
        mask2 = (mask2 > 0.5).float()
        mask3 = (mask3 > 0.5).float()
        mask4 = (mask4 > 0.5).float()

        return mask, mask2, mask3, mask4

    def get_mask_weight(self, mask):
        mask1, mask2, mask3, mask4 = self.generate_4mask(mask)

        mask1_balance = torch.ones_like(mask1)
        if (mask1 == 1).sum():
            mask1_balance[mask1 == 1] = 0.5 / ((mask1 == 1).sum().to(torch.float) / mask1.numel())
            mask1_balance[mask1 == 0] = 0.5 / ((mask1 == 0).sum().to(torch.float) / mask1.numel())

        mask2_balance = torch.ones_like(mask2)
        if (mask2 == 1).sum():
            mask2_balance[mask2 == 1] = 0.5 / ((mask2 == 1).sum().to(torch.float) / mask2.numel())
            mask2_balance[mask2 == 0] = 0.5 / ((mask2 == 0).sum().to(torch.float) / mask2.numel())

        mask3_balance = torch.ones_like(mask3)
        if (mask3 == 1).sum():
            mask3_balance[mask3 == 1] = 0.5 / ((mask3 == 1).sum().to(torch.float) / mask3.numel())
            mask3_balance[mask3 == 0] = 0.5 / ((mask3 == 0).sum().to(torch.float) / mask3.numel())

        mask4_balance = torch.ones_like(mask4)
        if (mask4 == 1).sum():
            mask4_balance[mask4 == 1] = 0.5 / ((mask4 == 1).sum().to(torch.float) / mask4.numel())
            mask4_balance[mask4 == 0] = 0.5 / ((mask4 == 0).sum().to(torch.float) / mask4.numel())

        return mask1_balance, mask2_balance, mask3_balance, mask4_balance

    def forward_features(self, image, *args, **kwargs):
        feat = self.FENet(image)
        return feat

    def forward(self, image, mask, label, edge_mask, *args, **kwargs):

        label = label.float()
        BCE_loss_full = nn.BCEWithLogitsLoss(reduction='none')

        # weight
        mask1, mask2, mask3, mask4 = self.generate_4mask(mask)
        mask1_balance, mask2_balance, mask3_balance, mask4_balance = self.get_mask_weight(mask)

        # forward
        feat = self.forward_features(image)

        pred_mask1, pred_mask2, pred_mask3, pred_mask4,feats = self.SegNet(feat)
        pred_mask = torch.sigmoid(pred_mask1)
        pred_logit = self.ClsNet(feat)
        pred_logit = torch.softmax(pred_logit, dim=1)
        pred_label = pred_logit[:, -1, ...]

        mask1_loss = torch.mean(BCE_loss_full(pred_mask1, mask1) * mask1_balance)
        mask2_loss = torch.mean(BCE_loss_full(pred_mask2, mask2) * mask2_balance)
        mask3_loss = torch.mean(BCE_loss_full(pred_mask3, mask3) * mask3_balance)
        mask4_loss = torch.mean(BCE_loss_full(pred_mask4, mask4) * mask4_balance)

        pre_edge = self.EdgeModule(feats)
        edge_loss = dice_loss(torch.sigmoid(pre_edge), edge_mask)

        seg_loss = mask1_loss + mask2_loss + mask3_loss + mask4_loss
        cls_loss = F.binary_cross_entropy(pred_label, label)
        combined_loss = 0.6 * seg_loss + 0.2 * cls_loss + 3 * edge_loss

        output_dict = {
            "backward_loss": combined_loss,
            "pred_mask": pred_mask,
            "pred_label": pred_label,

            "visual_loss": {
                "seg_loss": seg_loss,
                "cls_loss": cls_loss,
                "edge_loss": edge_loss,
                "combined_loss": combined_loss
            },
            "visual_image": {
                "pred_mask": pred_mask,
                "edge_mask": edge_mask
            }
        }

        return output_dict