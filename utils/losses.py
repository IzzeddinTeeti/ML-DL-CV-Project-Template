
import torch
import torch.nn as nn
import torch.nn.functional as F

class loss_ego_cls(nn.Module):
    def __init__(self):
        super(loss_ego_cls, self).__init__()
        self.criterion = nn.BCELoss()
        

    def forward(self, pred, target):
        '''
        pred: (batch, pred_len, num_ego_classes)
        target: (batch, pred_len, num_ego_classes)
        '''
        numc = pred.shape[-1]
        mask = target > -1
        masked_preds = pred[mask].reshape(-1, numc).to(pred.device)
        masked_labels = target[mask].reshape(-1)
        one_hot_labels = F.one_hot(masked_labels, num_classes=numc).float().to(pred.device)
        ego_loss = self.criterion (masked_preds, one_hot_labels)

        # pred = pred.view(-1, self.num_ego_classes)
        # target = target.view(-1, self.num_ego_classes)
        # loss = F.binary_cross_entropy(pred, target)
        # return loss
        # loss = F.binary_cross_entropy(pred, target)
        return ego_loss