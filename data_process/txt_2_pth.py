import os

import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm


def estimate_normals_with_open3d(points, k=4):
    # 将 numpy 数组转换为 Open3D 的点云格式
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 使用 Open3D 计算法向量
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))

    # 将法向量从 Open3D 格式转换回 numpy 数组
    normals = np.asarray(pcd.normals)

    return normals

def voxel_downsample_with_indices(points, voxel_size):
    """
    手动实现体素下采样，并返回下采样后的点及其在原始点云中的索引。

    参数:
    - points: 原始点云数据，Nx3的numpy数组。
    - voxel_size: 体素的大小。

    返回:
    - downsampled_points: 下采样后的点云，Mx3的numpy数组。
    - indices: 下采样后的点在原始点云中的索引，长度为M的列表。
    """
    # 将点云坐标转换为体素索引
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    # 生成唯一的体素坐标，并找到每个体素内第一个点的索引
    _, indices = np.unique(voxel_indices, axis=0, return_index=True)

    return indices

def txt_2_pth():
    split = "val"
    data_root = f"/home/keys/datasets/3d_huasheng/source/{split}"
    target_root = f"/home/keys/data1/3d_huasheng/source_0824/{split}"
    os.makedirs(target_root, exist_ok=True)
    scale = 1
    for item in os.listdir(data_root):
        data = np.loadtxt(os.path.join(data_root, item))
        xyz = data[:, :3] * scale
        # color = data[:, 3:6]
        pred = data[:, 3]
        segment = data[:, 4]
        scene_id = item[:-4]

        color = np.zeros_like(xyz)

        # segment2 = np.zeros_like(segment)
        # segment2[segment == 1] = 0
        # segment2[segment == 2] = 1
        # segment2[segment == 0] = -1
        normal = estimate_normals_with_open3d(xyz, k=32)
        # leaf_mask = segment == 2
        # instance = instance[leaf_mask]
        # show_pcd2(xyz, segment2)

        torch.save(
            {
                "coord": xyz,
                "color": color,
                "normal": normal,
                "semantic_gt": segment,
                "pred": pred,
                "scene_id": scene_id,
                # "superpoint": get_superpoint(group_plant_shuffled, normal)
            },
            os.path.join(target_root, item.replace(".txt", ".pth"))
        )

def txt_2_pth_inst():
    split = "train"
    data_root = f"/home/keys/datasets/3d_huasheng/source/{split}"
    target_root = f"/home/keys/data1/3d_huasheng/source_0822_inst/{split}"
    os.makedirs(target_root, exist_ok=True)
    scale = 0.1
    for item in os.listdir(data_root):
        data = np.loadtxt(os.path.join(data_root, item))
        xyz = data[:, :3] * scale
        color = data[:, 3:6]
        segment = data[:, 6]
        instance = data[:, 7]
        scene_id = item[:-4]

        # color = np.zeros_like(xyz)

        # segment2 = np.zeros_like(segment)
        # segment2[segment == 1] = 0
        # segment2[segment == 2] = 1
        # segment2[segment == 0] = -1
        leaf_mask = segment == 2
        xyz = xyz[leaf_mask]
        color = color[leaf_mask]
        instance = instance[leaf_mask]
        segment2 = np.zeros_like(instance)
        normal = estimate_normals_with_open3d(xyz, k=32)

        # show_pcd2(xyz, segment2)

        torch.save(
            {
                "coord": xyz,
                "color": color,
                "normal": normal,
                "semantic_gt": segment2,
                "instance_gt": instance,
                "scene_id": scene_id,
                # "superpoint": get_superpoint(group_plant_shuffled, normal)
            },
            os.path.join(target_root, item.replace(".txt", ".pth"))
        )


def txt_2_pth_no():
    split = "val"
    data_root = f"/home/yangxin/datasets/3d_huasheng/test_data/data"
    target_root = f"/data1/3d_huasheng/half_new_no_color/test"
    os.makedirs(target_root, exist_ok=True)
    scale = 0.1
    for item in os.listdir(data_root):
        data = np.loadtxt(os.path.join(data_root, item))
        xyz = data[:, :3] * scale
        # color = data[:, 3:6]
        # segment = data[:, 6]
        # instance = data[:, 7]
        scene_id = item[:-4]

        color = np.zeros_like(xyz)
        segment = np.zeros(xyz.shape[0])
        instance = np.zeros(xyz.shape[0])
        # segment2[segment == 1] = 0
        # segment2[segment == 2] = 1
        # segment2[segment == 0] = 2
        normal = estimate_normals_with_open3d(xyz, k=32)
        torch.save(
            {
                "coord": xyz,
                "color": color,
                "normal": normal,
                "semantic_gt": segment,
                "instance_gt": instance,
                "scene_id": scene_id,
                # "superpoint": get_superpoint(group_plant_shuffled, normal)
            },
            os.path.join(target_root, item.replace(".txt", ".pth"))
        )



if __name__ == "__main__":
    txt_2_pth()
    txt_2_pth_inst()
    txt_2_pth_no()
