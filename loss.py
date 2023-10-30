import torch
import torch.nn as nn
import torch.optim as optim

class ModelLoss(nn.Module):

    def __init__(self, params, device):
        super(ModelLoss, self).__init__()
        self.global_batch_size = params["batch_size"]
        self.small_addon_for_BCE = params["small_addon_for_BCE"]
        self.alpha_bce = params["alpha_bce"]
        self.beta_bce = params["beta_bce"]
        self.huber_delta = params["huber_delta"]
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.device = device

    def reg_loss_fn(self, reg_target, reg_pred, pos_equal_one_reg, pos_equal_one_sum):
        targ = reg_target * pos_equal_one_reg
        pred = reg_pred * pos_equal_one_reg
        loss = self.smooth_l1(targ, pred) / pos_equal_one_sum
        return torch.sum(loss) * (1.0 / self.global_batch_size)

    def prob_loss_fn(self, prob_pred, pos_equal_one, pos_equal_one_sum, neg_equal_one, neg_equal_one_sum):
        pos_log = torch.log(prob_pred + self.small_addon_for_BCE)
        pos_prod = -pos_equal_one * pos_log
        cls_pos_loss = pos_prod / pos_equal_one_sum

        neg_log = torch.log(1 - prob_pred + self.small_addon_for_BCE)
        neg_prod = -neg_equal_one * neg_log
        cls_neg_loss = neg_prod / neg_equal_one_sum

        cls_loss = torch.sum(self.alpha_bce * cls_pos_loss + self.beta_bce * cls_neg_loss) * (1.0 / self.global_batch_size)
        return cls_loss, torch.sum(cls_pos_loss) * (1.0 / self.global_batch_size), torch.sum(cls_neg_loss) * (1.0 / self.global_batch_size)

    def forward(self, reg_pred, prob_pred, targets, pos_equal_one, pos_equal_one_reg, pos_equal_one_sum, neg_equal_one, neg_equal_one_sum):
        reg_loss = self.reg_loss_fn(targets, reg_pred, pos_equal_one_reg, pos_equal_one_sum)
        cls_loss, cls_pos_loss, cls_neg_loss = self.prob_loss_fn(prob_pred, pos_equal_one, pos_equal_one_sum, neg_equal_one, neg_equal_one_sum)
        loss = reg_loss + cls_loss
        return loss, reg_loss, cls_loss, cls_pos_loss, cls_neg_loss
