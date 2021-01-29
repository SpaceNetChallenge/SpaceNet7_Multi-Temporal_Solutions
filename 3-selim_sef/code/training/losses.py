import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.modules.loss import BCEWithLogitsLoss


def dice_round(preds, trues, t=0.5):
    preds = (preds > t).float()
    return 1 - soft_dice_loss(preds, trues, reduce=False)


def soft_dice_loss(outputs, targets, per_image=False, reduce=True):
    batch_size = outputs.size()[0]
    eps = 1e-5
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union)
    if reduce:
        loss = loss.mean()

    return loss


def jaccard(outputs, targets, per_image=False, non_empty=False, min_pixels=5):
    batch_size = outputs.size()[0]
    eps = 1e-3
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    target_sum = torch.sum(dice_target, dim=1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    losses = 1 - (intersection + eps) / (torch.sum(dice_output + dice_target, dim=1) - intersection + eps)
    if non_empty:
        assert per_image == True
        non_empty_images = 0
        sum_loss = 0
        for i in range(batch_size):
            if target_sum[i] > min_pixels:
                sum_loss += losses[i]
                non_empty_images += 1
        if non_empty_images == 0:
            return 0
        else:
            return sum_loss / non_empty_images

    return losses.mean()


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return soft_dice_loss(input, target)


class LogCoshDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        x = soft_dice_loss(input, target)
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2)


class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.bce = BCEWithLogitsLoss()

    def forward(self, input, target):
        sigmoid_input = torch.sigmoid(input)
        return self.bce(input, target) + soft_dice_loss(sigmoid_input, target)


class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False, non_empty=False, apply_sigmoid=False,
                 min_pixels=5):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image
        self.non_empty = non_empty
        self.apply_sigmoid = apply_sigmoid
        self.min_pixels = min_pixels

    def forward(self, input, target):
        if self.apply_sigmoid:
            input = torch.sigmoid(input)
        return jaccard(input, target, per_image=self.per_image, non_empty=self.non_empty, min_pixels=self.min_pixels)


class ComboLoss(nn.Module):
    def __init__(self, weights, per_image=False, skip_empty=False, channel_weights=[1, 0.2, 0.1], channel_losses=None):
        super().__init__()
        self.weights = weights
        self.bce = BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.lcdice = LogCoshDiceLoss()
        self.jaccard = JaccardLoss(per_image=per_image)
        self.focal = BinaryFocalLoss()
        self.mapping = {'bce': self.bce,
                        'dice': self.dice,
                        'lcdice': self.lcdice,
                        'focal': self.focal,
                        'jaccard': self.jaccard}
        self.expect_sigmoid = {'dice', 'jaccard', 'lcdice'}
        self.per_channel = {'dice', 'jaccard', 'lcdice', "focal", "bce"}
        self.values = {}
        self.channel_weights = channel_weights
        self.channel_losses = channel_losses
        self.skip_empty = skip_empty

    def forward(self, outputs, targets):
        loss = 0
        weights = self.weights
        sigmoid_input = torch.sigmoid(outputs)
        for k, v in weights.items():
            if not v:
                continue
            val = 0
            if k in self.per_channel:
                channels = targets.size(1)
                for c in range(channels):
                    if not self.channel_losses or k in self.channel_losses[c]:
                        if self.skip_empty and torch.sum(targets[:, c, ...]) < 50:
                            continue
                        val += self.channel_weights[c] * self.mapping[k](
                            sigmoid_input[:, c, ...] if k in self.expect_sigmoid else outputs[:, c, ...],
                            targets[:, c, ...])

            else:
                val = self.mapping[k](sigmoid_input if k in self.expect_sigmoid else outputs, targets)

            self.values[k] = val
            loss += self.weights[k] * val
        return loss


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        eps = 1e-5
        non_ignored = targets.view(-1) != self.ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        outputs = torch.clamp(outputs, eps, 1. - eps)
        targets = torch.clamp(targets, eps, 1. - eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return (-(1. - pt) ** self.gamma * torch.log(pt)).mean()


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-6):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, inputs, targets):
        BCE_loss = binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


class EuclideanDistance(nn.Module):
    def __init__(self, only_target_pixels=True, ignore_value=255):
        super().__init__()
        self.ignore_value = ignore_value
        self.only_target_pixels = only_target_pixels

    def forward(self, input, target, mask=None):
        eps = 1e-16
        non_ignored = target[:, 0, ...].contiguous().view(-1) != self.ignore_value
        if self.only_target_pixels and mask is not None and torch.sum(mask) > 0:
            non_ignored &= mask.view(-1) > 0
        target_x = target[:, 1, ...].contiguous().view(-1)[non_ignored].contiguous()
        input_x = input[:, 1, ...].contiguous().view(-1)[non_ignored].contiguous()

        target_y = target[:, 0, ...].contiguous().view(-1)[non_ignored].contiguous()
        input_y = input[:, 0, ...].contiguous().view(-1)[non_ignored].contiguous()

        loss = torch.sqrt((target_x - input_x) ** 2 + (target_y - input_y) ** 2 + eps)
        return loss.mean()
