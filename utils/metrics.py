"""
This file will contain the metrics of the framework
COMPLETE_BOX_IOU_LOSS can be used as a loss function from 
https://pytorch.org/vision/stable/generated/torchvision.ops.complete_box_iou_loss.html
"""
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import MulticlassAveragePrecision
from torchmetrics.classification import MulticlassF1Score
import torch

class AverageMeter:
    """
    Class to be an average meter for any average metric like loss, accuracy, etc..
    """

    def __init__(self):
        self.momentum = 0.95 # 0 is better than 0.95
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if not self.momentum:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
        
        elif self.momentum:
            self.val = val
            if self.count == 0:
                self.avg = self.val
            else:
                self.avg = self.avg*self.momentum + (1-self.momentum)* val
            self.count += n
        
    # @property
    # def val(self):
    #     return self.avg


def mAP_ego(pred, gt, gurkirt_mAP=False):
    """
    Calculate the mAP of the model
    Gurkirt code devide the AP by the number of classes even if they have 0 or nan, the buil-in code find the 
    average of the non zero non nan APs
    """

    mask = gt != -1
    gt = gt[mask]
    pred = pred[mask]
    num_classes = pred.shape[1]
    
    if gurkirt_mAP:
        mAP = MulticlassAveragePrecision(num_classes=num_classes, average=None)
        res = mAP(pred, gt)
        res = torch.nansum(res) / num_classes
        return res
    else:
        mAP = MulticlassAveragePrecision(num_classes=num_classes, average="macro", thresholds=None)
        # print(pred, gt, mAP(pred, gt))
        # mAP.update(preds, gts)
        # print(gt[0], pred[0], mAP(pred, gt))
        return mAP(pred, gt)


def f1_ego(pred, gt):
    """
    Calculate the f1 score of the model
    """
    # print('shape', pred.shape, gt.shape)
    num_classes = pred.shape[2]
    # print('num_classes', num_classes)
    pred = pred.argmax(dim=2)
    # print('pred', pred.shape)
    # print(pred)
    # print(gt)
    mask = gt != -1
    gt = gt[mask]
    pred = pred[mask]
    f1 = MulticlassF1Score(num_classes=num_classes, average="macro", thresholds=None).to(pred.device)
    
    return f1(pred, gt)