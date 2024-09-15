"""
Misc Losses

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .builder import LOSSES
from pointcept.models.utils import offset2batch, batch2offset


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
            self,
            weight=None,
            size_average=None,
            reduce=None,
            reduction="mean",
            label_smoothing=0.0,
            loss_weight=1.0,
            ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight


@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(
            self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
                F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                * focal_weight
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                        torch.sum(
                            pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                        )
                        + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss


@LOSSES.register_module()
class MaskInstanceSegCriterion(nn.Module):
    def __init__(
            self,
            num_class,
            loss_weight=[1.0, 1.0, 1.0, 1.0],
            cost_weight=(1.0, 1.0, 1.0),
            ignore_index=-1):
        super(MaskInstanceSegCriterion, self).__init__()
        self.num_class = num_class
        self.ignore_index = ignore_index
        self.matcher = HungarianMatcher(cost_weight=cost_weight)
        loss_weight = torch.tensor(loss_weight)
        self.register_buffer('loss_weight', loss_weight)

    def forward(self, pred, target, **kwargs):
        instance = target['instance']
        segment = target['segment']
        offset = target['offset']

        gt_masks, gt_labels = get_insts_b(instance, segment, offset)
        target['gt_masks'] = gt_masks
        target['gt_labels'] = gt_labels

        [pred_classes, classes_offset] = pred['out_classes']
        [pred_scores, _] = pred['out_scores']
        [pred_masks, _] = pred['out_masks']

        offset = target['offset']

        point_batch = offset2batch(offset)
        batch_size = point_batch[-1] + 1
        classes_batch = offset2batch(classes_offset)

        device = offset.device

        indices = self.matcher(pred, target)

        idx = self._get_src_permutation_idx(indices)

        # class loss
        class_loss = torch.tensor(0, device=device, dtype=torch.float32)
        score_loss = torch.tensor(0, device=device, dtype=torch.float32)
        mask_bce_loss = torch.tensor(0, device=device, dtype=torch.float32)
        mask_dice_loss = torch.tensor(0, device=device, dtype=torch.float32)
        for b in range(batch_size):
            b_class_mask = classes_batch == b
            b_idx_mask = idx[0] == b
            b_tgt_class_o = gt_labels[b][indices[b][1]]

            b_pred_classes = pred_classes[b_class_mask]
            b_tgt_class = torch.full(
                [b_pred_classes.shape[0]],
                self.num_class - 1,
                dtype=torch.int64,
                device=device,
            )
            b_tgt_class[idx[1][b_idx_mask]] = b_tgt_class_o

            b_point_mask = point_batch == b
            b_pred_mask = pred_masks[b_point_mask].T[idx[1][b_idx_mask]]
            b_tgt_mask = gt_masks[b]

            with torch.no_grad():
                b_tgt_score = get_iou(b_pred_mask, b_tgt_mask).unsqueeze(1)
            b_pred_score = pred_scores[b_class_mask][idx[1][b_idx_mask]]

            class_loss += F.cross_entropy(b_pred_classes, b_tgt_class)
            score_loss += F.mse_loss(b_pred_score, b_tgt_score)
            mask_bce_loss += F.binary_cross_entropy_with_logits(b_pred_mask, b_tgt_mask.float())
            mask_dice_loss += dice_loss(b_pred_mask, b_tgt_mask.float())

        class_loss /= batch_size
        score_loss /= batch_size
        mask_bce_loss /= batch_size
        mask_bce_loss /= batch_size
        # print('class_loss: {}'.format(class_loss))
        # tgt_class_o = torch.cat([gt_labels[idx_gt] for gt_labels, (_, idx_gt) in zip(gt_labels, indices)])
        loss = (self.loss_weight[0] * class_loss + self.loss_weight[1] * mask_bce_loss +
                self.loss_weight[2] * mask_dice_loss + self.loss_weight[3] * score_loss)
        # return loss
        return dict(
            loss=loss,
            class_loss=class_loss,
            score_loss=score_loss,
            mask_bce_loss=mask_bce_loss,
            mask_dice_loss=mask_dice_loss,
        )

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


@LOSSES.register_module()
class OnlyMaskInstanceSegCriterion(nn.Module):
    def __init__(
            self,
            loss_weight=[1.0, 1.0],
            cost_weight=(1.0, 1.0),
            ignore_index=-1):
        super(OnlyMaskInstanceSegCriterion, self).__init__()
        self.ignore_index = ignore_index
        self.matcher = OnlyInstanceHungarianMatcher(cost_weight=cost_weight)
        loss_weight = torch.tensor(loss_weight)
        self.register_buffer('loss_weight', loss_weight)

    def forward(self, pred, target, **kwargs):
        instance = target['instance']
        segment = target['segment']
        offset = target['offset']
        loss_out = {}
        gt_masks, gt_labels = get_insts_b(instance, segment, offset)
        target['gt_masks'] = gt_masks
        target['gt_labels'] = gt_labels

        pred_masks = pred['out_masks']

        offset = target['offset']

        point_batch = offset2batch(offset)
        batch_size = point_batch[-1] + 1

        device = offset.device

        indices = self.matcher(pred, target)

        idx = self._get_src_permutation_idx(indices)

        # instance loss
        mask_bce_loss = torch.tensor(0, device=device, dtype=torch.float32)
        mask_dice_loss = torch.tensor(0, device=device, dtype=torch.float32)
        for b in range(batch_size):
            b_idx_mask = idx[0] == b

            b_point_mask = point_batch == b
            b_pred_mask = pred_masks[b_point_mask].T[idx[1][b_idx_mask]]
            b_tgt_mask = gt_masks[b]

            mask_bce_loss += F.binary_cross_entropy_with_logits(b_pred_mask, b_tgt_mask.float())
            mask_dice_loss += dice_loss(b_pred_mask, b_tgt_mask.float())

        mask_bce_loss /= batch_size
        mask_bce_loss /= batch_size
        # print('class_loss: {}'.format(class_loss))
        # tgt_class_o = torch.cat([gt_labels[idx_gt] for gt_labels, (_, idx_gt) in zip(gt_labels, indices)])
        loss = self.loss_weight[0] * mask_bce_loss + self.loss_weight[1] * mask_dice_loss
        loss_out['mask_bce_loss'] = mask_bce_loss
        loss_out['mask_dice_loss'] = mask_dice_loss
        if 'aux_outputs' in pred:
            for i, aux_outputs in enumerate(reversed(pred['aux_outputs'])):
                loss_i, loss_out_i = self.get_layer_loss(i, aux_outputs, target)
                loss += loss_i
                loss_out.update(loss_out_i)

        # return loss
        loss_out['loss'] = loss
        return loss_out

    def get_layer_loss(self, layer, aux_outputs, target):
        loss_out = {}
        instance = target['instance']
        segment = target['segment']
        target_N = instance.shape[0]

        next_cluster = aux_outputs['cluster']
        layer_mask_feat = aux_outputs['layer_mask_feat']
        aux_N = layer_mask_feat.shape[0]

        layer_pred = {}
        layer_target = {}
        if target_N == aux_N:
            gt_masks = target['gt_masks']
            # gt_labels = target['gt_labels']
            offset = target['offset']
            layer_target['gt_masks'] = gt_masks
            layer_target['offset'] = offset
            layer_pred['out_masks'] = layer_mask_feat
            # next target
            unique, cluster, counts = torch.unique(
                next_cluster, sorted=True, return_inverse=True, return_counts=True
            )
            idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
            # next_instance = instance[idx_ptr[:-1]]
            # next_segment = segment[idx_ptr[:-1]]
            #
            layer_batch = offset2batch(offset)
            layer_batch = layer_batch[idx_ptr[:-1]]
            layer_offset = batch2offset(layer_batch)
            target[f'{layer+1}_target'] = {
                'instance': instance[idx_ptr[:-1]],
                'segment': segment[idx_ptr[:-1]],
                'offset': layer_offset
            }

        else:
            layer_instance = target[f'{layer}_target']['instance']
            layer_segment = target[f'{layer}_target']['segment']
            layer_offset = target[f'{layer}_target']['offset']
            layer_gt_masks, layer_gt_labels = get_insts_b(layer_instance, layer_segment, layer_offset)
            layer_target['gt_masks'] = layer_gt_masks
            layer_target['offset'] = layer_offset
            layer_target['gt_labels'] = layer_gt_labels
            layer_pred['out_masks'] = layer_mask_feat

            # next target
            unique, cluster, counts = torch.unique(
                next_cluster, sorted=True, return_inverse=True, return_counts=True
            )
            idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
            # next_instance = instance[idx_ptr[:-1]]
            # next_segment = segment[idx_ptr[:-1]]
            #
            layer_batch = offset2batch(layer_offset)
            layer_batch = layer_batch[idx_ptr[:-1]]
            layer_offset = batch2offset(layer_batch)
            target[f'{layer + 1}_target'] = {
                'instance': instance[idx_ptr[:-1]],
                'segment': segment[idx_ptr[:-1]],
                'offset': layer_offset
            }


        layer_offset = layer_target['offset']
        layer_gt_masks = layer_target['gt_masks']
        layer_pred_masks = layer_pred['out_masks']

        point_batch = offset2batch(layer_offset)
        batch_size = point_batch[-1] + 1
        device = layer_offset.device

        indices = self.matcher(layer_pred, layer_target)

        idx = self._get_src_permutation_idx(indices)

        # instance loss
        mask_bce_loss = torch.tensor(0, device=device, dtype=torch.float32)
        mask_dice_loss = torch.tensor(0, device=device, dtype=torch.float32)
        for b in range(batch_size):
            b_idx_mask = idx[0] == b

            b_point_mask = point_batch == b
            b_pred_mask = layer_pred_masks[b_point_mask].T[idx[1][b_idx_mask]]
            b_tgt_mask = layer_gt_masks[b]

            mask_bce_loss += F.binary_cross_entropy_with_logits(b_pred_mask, b_tgt_mask.float())
            mask_dice_loss += dice_loss(b_pred_mask, b_tgt_mask.float())

        mask_bce_loss /= batch_size
        mask_bce_loss /= batch_size
        # print('class_loss: {}'.format(class_loss))
        # tgt_class_o = torch.cat([gt_labels[idx_gt] for gt_labels, (_, idx_gt) in zip(gt_labels, indices)])
        loss = self.loss_weight[0] * mask_bce_loss + self.loss_weight[1] * mask_dice_loss
        loss_out['mask_bce_loss'] = mask_bce_loss
        loss_out['mask_dice_loss'] = mask_dice_loss

        loss_out = {f'layer_{layer}_' + k: v for k, v in loss_out.items()}

        return loss, loss_out

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


def get_iou(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


@torch.jit.script
def batch_sigmoid_bce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: (num_querys, N)
        targets: (num_inst, N)
    Returns:
        Loss tensor
    """
    N = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')

    loss = torch.einsum('nc,mc->nm', pos, targets) + torch.einsum('nc,mc->nm', neg, (1 - targets))
    """
    问题出在torch.einsum('nc,mc->nm', neg, (1 - targets))，导致的 inf，采用 focalloss 的方式优化 batch_sigmoid_bce_loss
    """
    return loss / N


@torch.jit.script
def batch_sigmoid_bce_loss_scale(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: (num_querys, N)
        targets: (num_inst, N)
    Returns:
        Loss tensor
    """
    N = inputs.shape[1]

    scale = 0.5

    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none') * scale
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none') * scale

    loss = torch.einsum('nc,mc->nm', pos, targets) + torch.einsum('nc,mc->nm', neg, (1 - targets))
    """
    问题出在torch.einsum('nc,mc->nm', neg, (1 - targets))，导致的 inf，采用 focalloss 的方式优化 batch_sigmoid_bce_loss
    """
    return loss / N


# @torch.jit.script
def batch_sigmoid_bce_focal_loss1(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: 模型的输出，形状为 (num_queries, N)，N 是类别数目
        targets: 真实标签，形状为 (num_inst, N)，与 inputs 的 N 相同
        alpha: 类别权重，用于缓解类别不平衡
        gamma: 调节因子，用于降低易分类样本的权重
        reduction: 损失的降维方式，默认为 'none'
    Returns:
        Focal Loss 计算结果
    """
    alpha = 0.9
    gamma = 2.0
    N = inputs.shape[1]
    # 计算正类和负类的基本二元交叉熵损失
    pos_loss_base = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none')
    neg_loss_base = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')

    # 计算调制因子
    pos_prob = torch.sigmoid(inputs)
    # neg_prob = 1 - pos_prob
    pos_modulation = (1 - pos_prob) ** gamma
    neg_modulation = (pos_prob) ** gamma

    # 应用 Focal Loss 思想调整损失
    epsilon = 1e-8
    pos_loss = alpha * pos_modulation * (pos_loss_base + epsilon)
    neg_loss = (1 - alpha) * neg_modulation * (neg_loss_base + epsilon)
    # pos_loss = alpha * pos_modulation * pos_loss_base
    # neg_loss = (1 - alpha) * neg_modulation * neg_loss_base

    pos_loss1 = torch.einsum('nc,mc->nm', pos_loss, targets)
    neg_loss1 = torch.einsum('nc,mc->nm', neg_loss, (1 - targets))
    print(pos_loss1, neg_loss1)
    # 使用 einsum 应用目标调整，并合并正类和负类损失
    loss = pos_loss1 + neg_loss1
    return loss / N


@torch.jit.script
def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)  # 为什么这里是+1？
    return loss.mean()


@torch.jit.script
def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)  # 为什么这里是+1？
    return loss


def label_smoothing_fuc(labels, epsilon=0.1, classes=10):
    """
    对给定的标签应用标签平滑。

    参数:
        labels (torch.Tensor): one-hot 编码的标签，形状为 [classes, N]。
        epsilon (float): 平滑参数，决定平滑的强度。
        classes (int): 类别总数。

    返回:
        平滑后的标签 (torch.Tensor)。
    """
    # 确保 labels 是 one-hot 编码
    assert labels.dim() == 2 and labels.size(0) == classes
    # 计算平滑后的标签
    smoothed_labels = labels * (1 - epsilon) + (epsilon / classes)
    return smoothed_labels


def get_insts(instance_label, semantic_label, label_smoothing=True):
    """
    解决可能实例不连续的问题
    """
    device = instance_label.device
    num_insts = instance_label.max().item() + 1

    label_ids, masks = [], []

    for i in range(num_insts):
        i_inst_mask = instance_label == i
        if i_inst_mask.sum() == 0:
            continue  # 跳过不存在的实例
        sem_id = semantic_label[i_inst_mask].unique()
        if len(sem_id) > 1:
            # print(f"实例 {i} 有多个语义标签：{sem_id}")
            sem_id = sem_id[0]  # 假设选择第一个标签，需要根据实际情况调整
        label_ids.append(sem_id.item())
        masks.append(i_inst_mask)

    if len(masks) == 0:
        return None, None  # 没有有效的实例

    masks = torch.stack(masks).int()  # 将布尔掩码列表堆叠成张量，并转换为整数
    label_ids = torch.tensor(label_ids, device=device)

    # label_smoothing
    if label_smoothing:
        masks = label_smoothing_fuc(masks, classes=num_insts)
    return masks, label_ids


def get_insts_b(instance, segment, offset, label_smoothing=False):
    # instance = data_dict['instance']
    # segment = data_dict['segment']
    #
    # offset = data_dict['offset']

    point_batch = offset2batch(offset)
    batch_size = point_batch[-1] + 1

    gt_masks, gt_labels = [], []
    for b in range(batch_size):
        b_point_mask = point_batch == b
        b_gt_masks, b_gt_labels = get_insts(instance[b_point_mask], segment[b_point_mask],
                                            label_smoothing=label_smoothing)
        gt_masks.append(b_gt_masks)
        gt_labels.append(b_gt_labels)

    return gt_masks, gt_labels


class OnlyInstanceHungarianMatcher(nn.Module):
    def __init__(self, cost_weight):
        super(OnlyInstanceHungarianMatcher, self).__init__()
        self.cost_weight = torch.tensor(cost_weight)

    @torch.no_grad()
    def forward(self, pred, target):
        pred_masks = pred['out_masks']

        # instance = target['instance']
        # segment = target['segment']

        gt_masks = target['gt_masks']

        offset = target['offset']
        # num_queries = pred_masks.size(1)

        point_batch = offset2batch(offset)
        batch_size = point_batch[-1] + 1

        indices = []
        for b in range(batch_size):
            b_point_mask = point_batch == b

            # gt_masks, gt_labels = get_insts(instance[b_point_mask], segment[b_point_mask])
            b_gt_mask = gt_masks[b]

            pred_mask = pred_masks[b_point_mask].T  # [200, N]
            tgt_mask = b_gt_mask  # [num_ins, N]

            # cost_mask = batch_sigmoid_bce_loss(pred_mask, tgt_mask.float())
            # cost_mask = batch_sigmoid_bce_focal_loss(pred_mask, tgt_mask.float())
            cost_mask = batch_sigmoid_bce_loss_scale(pred_mask, tgt_mask.float())

            cost_dice = batch_dice_loss(pred_mask, tgt_mask.float())

            C = (self.cost_weight[1] * cost_mask + self.cost_weight[2] * cost_dice)
            C = C.cpu()

            indices.append(linear_sum_assignment(C))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class HungarianMatcher(nn.Module):
    def __init__(self, cost_weight):
        super(HungarianMatcher, self).__init__()

        self.cost_weight = torch.tensor(cost_weight)

    @torch.no_grad()
    def forward(self, pred, target):
        [pred_masks, _] = pred['out_masks']
        [pred_logits, logit_offset] = pred['out_classes']

        # instance = target['instance']
        # segment = target['segment']

        gt_labels = target['gt_labels']
        gt_masks = target['gt_masks']

        offset = target['offset']
        # num_queries = pred_masks.size(1)

        point_batch = offset2batch(offset)
        batch_size = point_batch[-1] + 1
        logit_batch = offset2batch(logit_offset)

        indices = []
        for b in range(batch_size):
            b_logit_mask = logit_batch == b
            b_point_mask = point_batch == b

            # gt_masks, gt_labels = get_insts(instance[b_point_mask], segment[b_point_mask])
            b_gt_mask = gt_masks[b]
            b_gt_labels = gt_labels[b]

            pred_prob = pred_logits[b_logit_mask].softmax(-1)
            tgt_idx = b_gt_labels
            cost_class = -pred_prob[:, tgt_idx]

            pred_mask = pred_masks[b_point_mask].T  # [200, N]
            tgt_mask = b_gt_mask  # [num_ins, N]

            # cost_mask = batch_sigmoid_bce_loss(pred_mask, tgt_mask.float())
            # cost_mask = batch_sigmoid_bce_focal_loss(pred_mask, tgt_mask.float())
            cost_mask = batch_sigmoid_bce_loss_scale(pred_mask, tgt_mask.float())

            cost_dice = batch_dice_loss(pred_mask, tgt_mask.float())

            C = (self.cost_weight[0] * cost_class + self.cost_weight[1] * cost_mask + self.cost_weight[2] * cost_dice)
            C = C.cpu()

            indices.append(linear_sum_assignment(C))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
