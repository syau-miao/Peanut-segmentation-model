"""
PointGroup for instance segmentation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Chengyao Wang
Please cite our work if the code is helpful to you.
"""

from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointgroup_ops import ballquery_batch_p, bfs_cluster
from sklearn.neighbors import KDTree

from pointcept.models.utils import offset2batch, batch2offset

from pointcept.models.builder import MODELS, build_model



@MODELS.register_module("PG-v2m3")
class PointGroup(nn.Module):
    def __init__(
            self,
            backbone,
            backbone_out_channels=64,
            semantic_num_classes=20,
            semantic_ignore_index=-1,
            segment_ignore_index=(-1, 0, 1),
            instance_ignore_index=-1,
            cluster_thresh=1.5,
            cluster_closed_points=300,
            cluster_propose_points=100,
            cluster_min_points=50,
            voxel_size=0.02,
            class_weights=None,
            test=False,
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
        self.test = test
        self.backbone = build_model(backbone)
        self.bias_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            norm_fn(backbone_out_channels),
            nn.ReLU(),
            nn.Linear(backbone_out_channels, 3),
        )

        self.seg_head = nn.Linear(backbone_out_channels, semantic_num_classes)

        self.ce_criteria = torch.nn.CrossEntropyLoss(ignore_index=semantic_ignore_index)

    def loss(self, bias_pred, data_dict):
        coord = data_dict["coord"]
        instance = data_dict["instance"]
        instance_centroid = data_dict["instance_centroid"]
        mask = (instance != self.instance_ignore_index).float()

        # mask = ~torch.isin(instance, torch.tensor(self.segment_ignore_index).to(instance.device))
        # mask = mask.to(torch.float)

        bias_gt = instance_centroid - coord
        # bias_dist = torch.sum(torch.abs(bias_pred - bias_gt), dim=-1)  # 使用z
        bias_dist = torch.sum(torch.abs(bias_pred[:, :2] - bias_gt[:, :2]), dim=-1)  # 忽略z轴
        bias_l1_loss = torch.sum(bias_dist * mask) / (torch.sum(mask) + 1e-8)

        bias_pred_norm = bias_pred / (
                torch.norm(bias_pred, p=2, dim=1, keepdim=True) + 1e-8
        )
        bias_gt_norm = bias_gt / (torch.norm(bias_gt, p=2, dim=1, keepdim=True) + 1e-8)
        cosine_similarity = -(bias_pred_norm * bias_gt_norm).sum(-1)
        bias_cosine_loss = torch.sum(cosine_similarity * mask) / (
                torch.sum(mask) + 1e-8
        )

        loss = bias_l1_loss + bias_cosine_loss
        return_dict = dict(
            loss=loss,
            bias_l1_loss=bias_l1_loss,
            bias_cosine_loss=bias_cosine_loss,
        )
        return return_dict

    def pred_feat(self, logit_pred, bias_pred, data_dict):
        coord = data_dict["coord"]
        offset = data_dict["offset"]
        center_pred = coord + bias_pred
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
            pred_classes=pred_classes,
            bias_pred=bias_pred,
        )
        return return_dict

    def pred_feat_m(self, bias_pred, data_dict):
        radius = self.cluster_thresh
        coord = data_dict["coord"]
        center_pred = coord + bias_pred

        # Convert to numpy for KDTree, if necessary keep on CPU to avoid GPU to CPU transfer costs
        center_pred_np = center_pred.detach().cpu().numpy()
        tree = KDTree(center_pred_np)

        cluster_mask = np.zeros(len(center_pred_np), dtype=int)
        i = 1
        v_pre = np.array([0, 0, 1])
        empty_t = 2
        while np.any(cluster_mask == 0):
            unclustered_indices = np.where(cluster_mask == 0)[0]
            p_index = unclustered_indices[np.argmin(center_pred_np[unclustered_indices, 2])]
            p = center_pred_np[p_index]
            empty_count = 0
            while True:
                within_radius = tree.query_radius([p], r=radius)[0]
                within_radius = within_radius[cluster_mask[within_radius] == 0]
                if empty_count >= empty_t:
                    break
                if within_radius.size == 0:
                    empty_count += 1
                    p += radius * v_pre
                    continue
                cluster_mask[within_radius] = i

                V = center_pred_np[within_radius] - p
                v_m = np.median(V, axis=0)
                norm = np.linalg.norm(v_m)
                if norm:
                    v_m /= norm
                v_m = 0.5 * v_pre + 0.5 * v_m
                norm = np.linalg.norm(v_m)
                if norm:
                    v_m /= norm
                v_pre = v_m
                p += radius * v_m

            i += 1

        # Validate clusters
        # valid_clusters, invalid_clusters = classify_clusters(cluster_mask, self.cluster_min_points)
        # # cluster_mask = redistribute_clusters(valid_clusters, invalid_clusters, coord.cpu().numpy(), cluster_mask)  # 将小于cluster_min_points的实例放分给最近的实例
        #
        # # # Update cluster_mask to only include valid clusters
        # cluster_mask = np.array([idx if idx in valid_clusters else 0 for idx in cluster_mask])  # 设为没有聚类的点
        # cluster_mask = redistribute_unassigned(coord.cpu().numpy(), cluster_mask)  # 将小于cluster_min_points的点 分给最近的实例

        valid_clusters = {j for j in range(1, i) if np.sum(cluster_mask == j) >= self.cluster_min_points}
        cluster_mask = np.array([0 if i not in valid_clusters else i for i in cluster_mask])

        unique_clusters = np.unique(cluster_mask)
        unique_clusters = unique_clusters[unique_clusters != 0]
        proposals_pred = torch.zeros((len(unique_clusters), len(cluster_mask)), dtype=torch.int32)

        for idx, cluster in enumerate(unique_clusters):
            cluster_indices = np.where(cluster_mask == cluster)[0]
            proposals_pred[idx, cluster_indices] = 1

        pred_scores = torch.tensor([proposals_pred[idx].float().mean().item() for idx in range(len(unique_clusters))])
        pred_classes = torch.zeros(len(unique_clusters), dtype=torch.int32)

        return_dict = dict(
            pred_scores=pred_scores,
            pred_masks=proposals_pred,
            pred_classes=pred_classes,
            bias_pred=bias_pred,
        )

        return return_dict

    def mask_bias(self, bias_pred, data_dict, radius=0.05):
        coord = data_dict["coord"]
        # instance = data_dict["instance"]
        instance_bottom = data_dict["instance_bottom"]
        instance_stem_v = data_dict["instance_stem_v"]

        center_pred = coord + bias_pred
        coord_B_v = center_pred - instance_bottom
        bias_v = (coord_B_v * instance_stem_v).sum(1, keepdim=True) * instance_stem_v - coord_B_v
        bias_v_norm = torch.norm(bias_v, dim=1)
        out_center_mask = bias_v_norm > radius
        return out_center_mask

    def mask_loss(self, bias_pred, data_dict, out_mask):
        coord = data_dict["coord"][out_mask]
        # offset = data_dict["offset"]
        instance = data_dict["instance"][out_mask]
        mask = (instance != self.instance_ignore_index).float()

        # instance_centroid = data_dict["instance_centroid"]
        instance_bottom = data_dict["instance_bottom"][out_mask]
        instance_stem_v = data_dict["instance_stem_v"][out_mask]
        coord_B_v = coord - instance_bottom
        bias_gt = (coord_B_v * instance_stem_v).sum(1, keepdim=True) * instance_stem_v - coord_B_v

        bias_dist = torch.sum(torch.abs(bias_pred - bias_gt), dim=-1)
        bias_l1_loss = torch.sum(bias_dist * mask) / (torch.sum(mask) + 1e-8)

        bias_pred_norm = bias_pred / (
                torch.norm(bias_pred, p=2, dim=1, keepdim=True) + 1e-8
        )
        bias_gt_norm = bias_gt / (torch.norm(bias_gt, p=2, dim=1, keepdim=True) + 1e-8)
        cosine_similarity = -(bias_pred_norm * bias_gt_norm).sum(-1)
        bias_cosine_loss = torch.sum(cosine_similarity * mask) / (
                torch.sum(mask) + 1e-8
        )

        loss = bias_l1_loss + bias_cosine_loss
        return_dict = dict(
            out_loss=loss,
            out_bias_l1_loss=bias_l1_loss,
            out_bias_cosine_loss=bias_cosine_loss,
        )
        return return_dict

    def forward(self, data_dict):
        Point = self.backbone(data_dict)
        feat = Point.feat
        bias_pred = self.bias_head(feat)
        return_dict = dict()
        if not self.test:
            return_dict = self.loss(bias_pred, data_dict)
        if not self.training:
            bias_pred[:, 2] = 0
            pred_dict = self.pred_feat_m(bias_pred, data_dict)
            return_dict.update(pred_dict)
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
