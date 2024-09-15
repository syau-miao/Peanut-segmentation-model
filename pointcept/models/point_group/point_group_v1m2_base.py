"""
PointGroup for instance segmentation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Chengyao Wang
Please cite our work if the code is helpful to you.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointgroup_ops import ballquery_batch_p, bfs_cluster
from pointcept.models.utils import offset2batch, batch2offset

from pointcept.models.builder import MODELS, build_model


class SEBlock1D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock1D, self).__init__()
        self.in_channels = in_channels
        # 全连接层代替原来的卷积层
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 假设x的形状为[N, C]，其中N是特征点数，C是通道数
        # 全局平均池化，计算每个通道的平均值
        global_avg = torch.mean(x, dim=0, keepdim=True)  # 形状为[1, C]
        # 计算通道注意力权重
        scale = self.excitation(global_avg)  # 形状为[1, C]
        # 将注意力权重应用到原始特征上
        return x * scale.expand_as(x)


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)

        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)

        return out + x


class TinyNet1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TinyNet1D, self).__init__()
        # 用于初始特征提取的线性层
        self.linear1 = nn.Linear(in_channels, out_channels)
        # 使用SEBlock1D作为注意力机制
        self.se_block = SEBlock1D(out_channels)
        # 第二个线性层用于进一步特征提取
        self.linear2 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.se_block(x)
        x = F.relu(self.linear2(x))
        return x


@MODELS.register_module("PG-v1m2")
class PointGroup(nn.Module):
    def __init__(
            self,
            backbone,
            backbone_out_channels=64,
            semantic_num_classes=20,
            semantic_ignore_index=-1,
            segment_ignore_index=(-1, 0, 1),
            instance_ignore_index=(-1, 0),
            cluster_thresh=1.5,
            cluster_closed_points=300,
            cluster_propose_points=100,
            cluster_min_points=50,
            voxel_size=0.02,
            class_weights=None,
            organ_semantic_num_classes=20,
            organ_class_weights=None,
            organ_segment_ignore_index=(-1, 0)
    ):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.semantic_num_classes = semantic_num_classes
        self.segment_ignore_index = segment_ignore_index
        self.semantic_ignore_index = semantic_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.cluster_thresh = cluster_thresh
        self.cluster_closed_points = cluster_closed_points
        self.cluster_propose_points = cluster_propose_points
        self.cluster_min_points = cluster_min_points
        self.voxel_size = voxel_size
        self.class_weights = torch.tensor(class_weights) if class_weights is not None \
            else torch.ones(semantic_num_classes)
        self.organ_semantic_num_classes = organ_semantic_num_classes
        self.organ_class_weights = torch.tensor(organ_class_weights) if organ_class_weights is not None \
            else torch.ones(organ_semantic_num_classes)
        self.organ_segment_ignore_index = organ_segment_ignore_index

        self.backbone = build_model(backbone)

        mid_channels = backbone_out_channels // 2

        self.single_feat_net = TinyNet1D(in_channels=backbone_out_channels, out_channels=mid_channels)
        self.organ_feat_net = TinyNet1D(in_channels=backbone_out_channels, out_channels=mid_channels)

        self.bias_head = nn.Sequential(
            nn.Linear(mid_channels, mid_channels),
            norm_fn(mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, 3),
        )
        self.seg_head = nn.Linear(mid_channels, semantic_num_classes)

        self.organ_bias_head = nn.Sequential(
            nn.Linear(mid_channels, mid_channels),
            norm_fn(mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, 3),
        )
        self.organ_seg_head = nn.Linear(mid_channels, organ_semantic_num_classes)

        self.organ_to_single_bias_head = nn.Sequential(
            nn.Linear(mid_channels, mid_channels),
            norm_fn(mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, 3),
        )

        self.ce_criteria = torch.nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=semantic_ignore_index)
        self.organ_ce_criteria = torch.nn.CrossEntropyLoss(weight=self.organ_class_weights,
                                                           ignore_index=semantic_ignore_index)

    def compute_ins_loss(self, bias_pred, coord, instance, instance_centroid, segment_ignore_index):
        # mask = ~torch.isin(instance, torch.tensor(segment_ignore_index).to(instance.device))
        # mask = mask.to(torch.float)
        mask = (instance != self.instance_ignore_index).float()
        bias_gt = instance_centroid - coord
        bias_dist = torch.sum(torch.abs(bias_pred - bias_gt), dim=-1)
        bias_l1_loss = torch.sum(bias_dist * mask) / (torch.sum(mask) + 1e-8)

        bias_pred_norm = bias_pred / (torch.norm(bias_pred, p=2, dim=1, keepdim=True) + 1e-8)
        bias_gt_norm = bias_gt / (torch.norm(bias_gt, p=2, dim=1, keepdim=True) + 1e-8)
        cosine_similarity = -(bias_pred_norm * bias_gt_norm).sum(-1)
        bias_cosine_loss = torch.sum(cosine_similarity * mask) / (torch.sum(mask) + 1e-8)

        return bias_l1_loss, bias_cosine_loss

    def loss(self, logit_pred, bias_pred, organ_logit_pred, organ_bias_pred, organ_to_single_bias_pred, data_dict):
        # group loss
        coord = data_dict["coord"]
        # offset = data_dict["offset"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        instance_centroid = data_dict["instance_centroid"]

        seg_loss = self.ce_criteria(logit_pred, segment)

        bias_l1_loss, bias_cosine_loss = self.compute_ins_loss(
            bias_pred=bias_pred, coord=coord, instance=instance,
            instance_centroid=instance_centroid,
            segment_ignore_index=self.segment_ignore_index)

        group_loss = seg_loss + bias_l1_loss + bias_cosine_loss
        group_return_dict = dict(
            group_loss=group_loss,
            group_seg_loss=seg_loss,
            group_bias_l1_loss=bias_l1_loss,
            group_bias_cosine_loss=bias_cosine_loss,
        )

        return_dict = dict(
            loss=group_loss,
            group_seg_loss=seg_loss,
            group_bias_l1_loss=bias_l1_loss,
            group_bias_cosine_loss=bias_cosine_loss,
        )
        return return_dict

        # organ loss
        organ_segment = data_dict["organ_segment"]
        organ_instance = data_dict["organ_instance"]
        organ_instance_centroid = data_dict["organ_instance_centroid"]
        organ_seg_loss = self.organ_ce_criteria(organ_logit_pred, organ_segment)

        organ_bias_l1_loss, organ_bias_cosine_loss = self.compute_ins_loss(
            bias_pred=organ_bias_pred, coord=coord, instance=organ_instance,
            instance_centroid=organ_instance_centroid,
            segment_ignore_index=self.organ_segment_ignore_index)
        organ_loss = organ_seg_loss + organ_bias_l1_loss + organ_bias_cosine_loss

        organ_return_dict = dict(
            organ_loss=organ_loss,
            organ_seg_loss=organ_seg_loss,
            organ_bias_l1_loss=organ_bias_l1_loss,
            organ_bias_cosine_loss=organ_bias_cosine_loss,
        )

        # organ_to_single loss
        """
        bias_gt = instance_centroid - coord
        organ_to_single_gt = instance_centroid - organ_instance_centroid
        organ_instance_centroid_pred = organ_bias_pred + coord
        instance_centroid = organ_to_single_bias_pred + organ_instance_centroid_pred 
        """
        organ_to_single_bias_l1_loss, organ_to_single_bias_cosine_loss = self.compute_ins_loss(
            bias_pred=organ_to_single_bias_pred, coord=organ_instance_centroid, instance=organ_instance,
            instance_centroid=instance_centroid,
            segment_ignore_index=self.organ_segment_ignore_index)

        organ_to_single_loss = organ_to_single_bias_l1_loss + organ_to_single_bias_cosine_loss
        organ_to_single_return_dict = dict(
            organ_to_single_loss=organ_to_single_loss,
            organ_to_single_bias_l1_loss=organ_to_single_bias_l1_loss,
            organ_to_single_bias_cosine_loss=organ_to_single_bias_cosine_loss,
        )

        loss = group_loss + 0.5 * organ_loss + 0.5 * organ_to_single_loss

        return_dict = dict(
            loss=loss
        )
        return_dict.update(group_return_dict)
        return_dict.update(organ_return_dict)
        return_dict.update(organ_to_single_return_dict)
        return return_dict

    def process_feat(self, logit_pred, bias_pred, coord, offset, segment_ignore_index, cluster_thresh,
                     cluster_closed_points, cluster_min_points, cluster_propose_points):
        center_pred = coord + bias_pred
        center_pred /= self.voxel_size
        logit_pred = F.softmax(logit_pred, dim=-1)
        segment_pred = torch.max(logit_pred, 1)[1]  # [n]
        # cluster
        mask = (
            ~torch.concat(
                [
                    (segment_pred == index).unsqueeze(-1)
                    for index in segment_ignore_index
                ],
                dim=1,
            )
            .sum(-1)
            .bool()
        )

        if mask.sum() == 0:
            proposals_idx = torch.zeros(0).int()
            proposals_offset = torch.zeros(1).int()
        else:
            center_pred_ = center_pred[mask]
            segment_pred_ = segment_pred[mask]

            batch_ = offset2batch(offset)[mask]
            offset_ = nn.ConstantPad1d((1, 0), 0)(batch2offset(batch_))
            idx, start_len = ballquery_batch_p(
                center_pred_,
                batch_.int(),
                offset_.int(),
                cluster_thresh,
                cluster_closed_points,
            )
            proposals_idx, proposals_offset = bfs_cluster(
                segment_pred_.int().cpu(),
                idx.cpu(),
                start_len.cpu(),
                cluster_min_points,
            )
            proposals_idx[:, 1] = (
                mask.nonzero().view(-1)[proposals_idx[:, 1].long()].int()
            )

        # get proposal
        proposals_pred = torch.zeros(
            (proposals_offset.shape[0] - 1, center_pred.shape[0]), dtype=torch.int
        )
        proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
        instance_pred = segment_pred[
            proposals_idx[:, 1][proposals_offset[:-1].long()].long()
        ]
        proposals_point_num = proposals_pred.sum(1)
        proposals_mask = proposals_point_num > cluster_propose_points
        proposals_pred = proposals_pred[proposals_mask]
        instance_pred = instance_pred[proposals_mask]

        pred_scores = []
        pred_classes = []
        pred_masks = proposals_pred.detach().cpu()
        for proposal_id in range(len(proposals_pred)):
            segment_ = proposals_pred[proposal_id]
            confidence_ = logit_pred[
                segment_.bool(), instance_pred[proposal_id]
            ].mean()
            object_ = instance_pred[proposal_id]
            pred_scores.append(confidence_)
            pred_classes.append(object_)
        if len(pred_scores) > 0:
            pred_scores = torch.stack(pred_scores).cpu()
            pred_classes = torch.stack(pred_classes).cpu()
        else:
            pred_scores = torch.tensor([])
            pred_classes = torch.tensor([])

        return pred_scores, pred_masks, pred_classes

    def pred_feat(self, logit_pred, bias_pred, organ_logit_pred, organ_bias_pred, organ_to_single_bias_pred,
                      data_dict):
        # group
        coord = data_dict['coord']
        offset = data_dict['offset']
        pred_scores, pred_masks, pred_classes = self.process_feat(
            logit_pred=logit_pred, bias_pred=bias_pred,
            coord=coord, offset=offset,
            segment_ignore_index=self.segment_ignore_index,
            cluster_thresh=self.cluster_thresh,
            cluster_closed_points=self.cluster_closed_points,
            cluster_min_points=self.cluster_min_points,
            cluster_propose_points=self.cluster_propose_points)

        # organ
        organ_pred_scores, organ_pred_masks, organ_pred_classes = self.process_feat(
            logit_pred=organ_logit_pred, bias_pred=organ_bias_pred,
            coord=coord, offset=offset,
            segment_ignore_index=self.organ_segment_ignore_index,
            cluster_thresh=self.cluster_thresh / 5,
            cluster_closed_points=int(self.cluster_closed_points / 5),
            cluster_min_points=self.cluster_min_points,
            cluster_propose_points=self.cluster_propose_points,
            )

        # organ_to_single

        return_dict = dict(
            pred_scores=pred_scores,
            pred_masks=pred_masks,
            pred_classes=pred_classes
        )
        return return_dict

    def pred_feat_1(self, logit_pred, bias_pred, organ_logit_pred, organ_bias_pred, organ_to_single_bias_pred, data_dict):

        coord = data_dict['coord']
        offset = data_dict['offset']

        # group
        center_pred = coord + bias_pred
        # organ
        # organ_center_pred = coord + organ_bias_pred
        # single_center_pred = organ_to_single_bias_pred + organ_center_pred
        # center_pred = (center_pred + single_center_pred) / 2

        center_pred /= self.voxel_size
        logit_pred = F.softmax(logit_pred, dim=-1)
        segment_pred = torch.max(logit_pred, 1)[1]  # [n]
        # cluster
        mask = (
            ~torch.concat(
                [
                    (segment_pred == index).unsqueeze(-1)
                    for index in self.segment_ignore_index
                ],
                dim=1,
            )
            .sum(-1)
            .bool()
        )

        if mask.sum() == 0:
            proposals_idx = torch.zeros(0).int()
            proposals_offset = torch.zeros(1).int()
        else:
            center_pred_ = center_pred[mask]
            segment_pred_ = segment_pred[mask]

            batch_ = offset2batch(offset)[mask]
            offset_ = nn.ConstantPad1d((1, 0), 0)(batch2offset(batch_))
            idx, start_len = ballquery_batch_p(
                center_pred_,
                batch_.int(),
                offset_.int(),
                self.cluster_thresh,
                self.cluster_closed_points,
            )
            proposals_idx, proposals_offset = bfs_cluster(
                segment_pred_.int().cpu(),
                idx.cpu(),
                start_len.cpu(),
                self.cluster_min_points,
            )
            proposals_idx[:, 1] = (
                mask.nonzero().view(-1)[proposals_idx[:, 1].long()].int()
            )

        # get proposal
        proposals_pred = torch.zeros(
            (proposals_offset.shape[0] - 1, center_pred.shape[0]), dtype=torch.int
        )
        proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
        instance_pred = segment_pred[
            proposals_idx[:, 1][proposals_offset[:-1].long()].long()
        ]
        proposals_point_num = proposals_pred.sum(1)
        proposals_mask = proposals_point_num > self.cluster_propose_points
        proposals_pred = proposals_pred[proposals_mask]
        instance_pred = instance_pred[proposals_mask]

        pred_scores = []
        pred_classes = []
        pred_masks = proposals_pred.detach().cpu()
        for proposal_id in range(len(proposals_pred)):
            segment_ = proposals_pred[proposal_id]
            confidence_ = logit_pred[
                segment_.bool(), instance_pred[proposal_id]
            ].mean()
            object_ = instance_pred[proposal_id]
            pred_scores.append(confidence_)
            pred_classes.append(object_)
        if len(pred_scores) > 0:
            pred_scores = torch.stack(pred_scores).cpu()
            pred_classes = torch.stack(pred_classes).cpu()
        else:
            pred_scores = torch.tensor([])
            pred_classes = torch.tensor([])

        return_dict = dict(
            pred_scores=pred_scores,
            pred_masks=pred_masks,
            pred_classes=pred_classes
        )
        return return_dict

    def forward(self, data_dict):
        feat = self.backbone(data_dict)
        single_feat = self.single_feat_net(feat)
        organ_feat = self.organ_feat_net(feat)

        bias_pred = self.bias_head(single_feat)
        logit_pred = self.seg_head(single_feat)

        organ_bias_pred = self.organ_bias_head(organ_feat)
        organ_logit_pred = self.organ_seg_head(organ_feat)

        organ_to_single_bias_pred = self.organ_to_single_bias_head(organ_feat)

        # compute loss
        return_dict = self.loss(
            logit_pred, bias_pred, organ_logit_pred, organ_bias_pred, organ_to_single_bias_pred, data_dict)

        if not self.training:
            # pred
            out_dict = self.pred_feat(
                logit_pred, bias_pred, organ_logit_pred, organ_bias_pred, organ_to_single_bias_pred, data_dict)
            return_dict.update(out_dict)
        return return_dict

# import open3d as o3d
# import numpy as np
# def show_pcd1(data):
#
#     xyz = data.copy()
#     # 创建第一个点云对象
#     pcd1 = o3d.geometry.PointCloud()
#     pcd1.points = o3d.utility.Vector3dVector(xyz)
#
#     # max_z = abs(np.max(xyz[:,2]))
#     # 创建坐标系几何对象
#     coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
#     o3d.visualization.draw_geometries([pcd1, coord_frame], )
#
#
# def show_pcd2(xyz, l):
#     colors_txt = \
#     open('/home/yangxin/workspace/Pointcept-1.5.1/project/group_seg/data_process/data_show/colors.txt').readlines()[
#         0].split(';')
#     colors_txt = colors_txt * 100
#     label_to_color = {}
#     for index, item in enumerate(colors_txt):
#         i_color = item.strip().split(' ')
#         label_to_color[index] = np.array([float(i) for i in i_color])
#
#     label_to_color[-1] = [0, 0, 0]
#
#     point_colors = [label_to_color[int(label)] for label in l]
#
#     # 创建第一个点云对象
#     pcd1 = o3d.geometry.PointCloud()
#     pcd1.points = o3d.utility.Vector3dVector(xyz)
#     pcd1.colors = o3d.utility.Vector3dVector(point_colors)
#
#     # max_z = abs(np.max(xyz[:, 2]))
#     # 创建坐标系几何对象
#     # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max_z / 4)
#
#     o3d.visualization.draw_geometries([pcd1], )  # z轴朝上