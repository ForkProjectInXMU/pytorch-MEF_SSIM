#%%
import sys
import cv2
import os
import numpy as np
import torch
from pytorch_mef_ssim.MEF_SSIM import mef_ssim


if __name__ == '__main__':
    oe_path = '../oe.png'
    ue_path = '../ue.png'
    gt_path = '../gt.png'
    oe = cv2.imread(oe_path, 0).astype(np.float64)
    ue = cv2.imread(ue_path, 0).astype(np.float64)
    img_seq = torch.Tensor(np.stack([oe, ue], 0)).unsqueeze(0)
    gt = cv2.imread(gt_path, 0).astype(np.float64)
    gt = np.expand_dims(gt, 0)
    gt = np.expand_dims(gt, 0)
    gt = torch.Tensor(gt)

    print(mef_ssim(img_seq, gt))

