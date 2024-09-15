#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/9 18:22
# @Author  : yangxin
# @Email   : yangxinnc@163.com
# @File    : corn3d_ins.py
# @Project : PlantPointSeg
# @Software: PyCharm
import glob
import os
from copy import deepcopy

import numpy as np
import torch
from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class Corn3dGroupDataset(DefaultDataset):
    # class2id = np.array([0, 1, 2, 3])

    def __init__(
            self,
            split="train",
            data_root="data/corn3d",
            transform=None,
            test_mode=False,
            test_cfg=None,
            loop=1,
            ignore_index=-1,
    ):
        self.ignore_index = ignore_index
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        data = torch.load(data_path)
        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        scene_id = data["scene_id"]
        if "semantic_gt" in data.keys():
            segment = data["semantic_gt"].reshape([-1]).astype(int)
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1]).astype(int)
        else:
            instance = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
        )
        if "superpoint" in data.keys():
            superpoint = data["superpoint"]
            data_dict["superpoint"] = superpoint

        return data_dict


@DATASETS.register_module()
class Corn3dGroupDatasetV2(DefaultDataset):
    # class2id = np.array([0, 1, 2, 3])

    def __init__(
            self,
            split="train",
            data_root="data/corn3d",
            transform=None,
            test_mode=False,
            test_cfg=None,
            loop=1,
            ignore_index=-1,
    ):
        self.ignore_index = ignore_index
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def _gen_centroid_bottom(self, instance, instance_centroid, instance_bottom):
        centroids = np.ones((instance.shape[0], 3))
        bottoms = np.ones((instance.shape[0], 3))
        for inst in np.unique(instance):
            i_mask = (instance == inst)
            centroids[i_mask] = instance_centroid[inst]
            bottoms[i_mask] = instance_bottom[inst]
        return centroids, bottoms

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        data = torch.load(data_path)
        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        # scene_id = data["scene_id"]
        if "semantic_gt" in data.keys():
            segment = data["semantic_gt"].reshape([-1]).astype(int)
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1]).astype(int)
        else:
            instance = np.ones(coord.shape[0]) * -1
        if "organ_semantic_gt" in data.keys():
            organ_segment = data["organ_semantic_gt"].reshape([-1]).astype(int)
        else:
            organ_segment = np.ones(coord.shape[0]) * -1
        if "organ_instance_gt" in data.keys():
            organ_instance = data["organ_instance_gt"].reshape([-1]).astype(int)
        else:
            organ_instance = np.ones(coord.shape[0]) * -1

        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            instance=instance,
            organ_segment=organ_segment,
            organ_instance=organ_instance,
            name=self.get_data_name(idx),
        )
        if "superpoint" in data.keys():
            superpoint = data["superpoint"]
            data_dict["superpoint"] = superpoint

        return data_dict


@DATASETS.register_module()
class Corn3dGroupDatasetV3(DefaultDataset):
    # class2id = np.array([0, 1, 2, 3])

    def __init__(
            self,
            split="train",
            data_root="data/corn3d",
            transform=None,
            test_mode=False,
            test_cfg=None,
            loop=1,
            ignore_index=-1,
    ):
        self.ignore_index = ignore_index
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )


    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        data = torch.load(data_path)
        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        # scene_id = data["scene_id"]
        if "semantic_gt" in data.keys():
            segment = data["semantic_gt"].reshape([-1]).astype(int)
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "plant_semantic_gt" in data.keys():
            instance = data["plant_semantic_gt"].reshape([-1]).astype(int)
        else:
            instance = np.ones(coord.shape[0]) * -1
        if "plant_instance_gt" in data.keys():
            organ_segment = data["plant_instance_gt"].reshape([-1]).astype(int)
        else:
            organ_segment = np.ones(coord.shape[0]) * -1

        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            instance=instance,
            organ_segment=organ_segment,
            name=self.get_data_name(idx),
        )


        return data_dict


@DATASETS.register_module()
class Corn3dOrganDataset(DefaultDataset):
    # class2id = np.array([0, 1, 2, 3])

    def __init__(
            self,
            split="train",
            data_root="data/corn3d",
            transform=None,
            test_mode=False,
            test_cfg=None,
            loop=1,
            ignore_index=-1,
    ):
        self.ignore_index = ignore_index
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        data = torch.load(data_path)
        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        scene_id = data["scene_id"]
        if "semantic_gt" in data.keys():
            segment = data["semantic_gt"].reshape([-1]).astype(int)
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1]).astype(int)
        else:
            instance = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
        )
        if "superpoint" in data.keys():
            superpoint = data["superpoint"]
            data_dict["superpoint"] = superpoint

        return data_dict


@DATASETS.register_module()
class Corn3dGroupSemanticDataset(DefaultDataset):
    # class2id = np.array([0, 1, 2, 3])

    def __init__(
            self,
            split="train",
            data_root="data/corn3d",
            transform=None,
            test_mode=False,
            test_cfg=None,
            loop=1,
            ignore_index=-1,
    ):
        self.ignore_index = ignore_index
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        data = torch.load(data_path)
        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        scene_id = data["scene_id"]
        if "semantic_gt" in data.keys():
            segment = data["semantic_gt"].reshape([-1]).astype(int)
        else:
            segment = np.ones(coord.shape[0]) * -1

        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            scene_id=scene_id,
        )
        return data_dict


@DATASETS.register_module()
class Corn3dGroupSemanticDataset2(DefaultDataset):
    # class2id = np.array([0, 1, 2, 3])

    def __init__(
            self,
            split="train",
            data_root="data/corn3d",
            transform=None,
            test_mode=False,
            test_cfg=None,
            loop=1,
            ignore_index=-1,
    ):
        self.ignore_index = ignore_index
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        data = torch.load(data_path)
        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        scene_id = data["scene_id"]
        if "semantic_gt" in data.keys():
            segment = data["semantic_gt"].reshape([-1]).astype(int)
        else:
            segment = np.ones(coord.shape[0]).astype(int) * -1
        segment1 = np.zeros(coord.shape[0]).astype(int)

        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment1,
            instance=segment,
            scene_id=scene_id,
            name=self.get_data_name(idx),
        )
        return data_dict

@DATASETS.register_module()
class Corn3dGroupSemanticSPDataset(DefaultDataset):
    # class2id = np.array([0, 1, 2, 3])

    def __init__(
            self,
            split="train",
            data_root="data/corn3d",
            transform=None,
            test_mode=False,
            test_cfg=None,
            loop=1,
            ignore_index=-1,
    ):
        self.ignore_index = ignore_index
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        data = torch.load(data_path)
        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        scene_id = data["scene_id"]
        if "semantic_gt" in data.keys():
            segment = data["semantic_gt"].reshape([-1]).astype(int)
        else:
            segment = np.ones(coord.shape[0]) * -1

        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            scene_id=scene_id,
        )
        filename_split = os.path.split(data_path)
        sp_filename = os.path.join(filename_split[0][:filename_split[0].rfind('/')], 'sp', filename_split[1])
        superpoint = torch.load(sp_filename)
        data_dict["superpoint"] = superpoint
        return data_dict