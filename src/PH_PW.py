import math

import numpy as np
import os
import math
import open3d as o3d

def P_HW(data):
    '''

    :param data: Plant point cloud data
    :return:Plant height and width
    '''
    xyzl=data
    print(xyzl)
    points = xyzl[xyzl[:, -2] == 1]
    points = points[:, :3]
    max_z = np.max(points[:, 2])
    min_z = np.min(points[:, 2])
    z2 = (max_z - min_z) / 100 + min_z
    dataz2 = points[points[:, 2] < z2]
    x = dataz2[:, 0]
    y = dataz2[:, 1]
    z = dataz2[:, 2] - dataz2[:, 2]
    project_points = np.c_[x, y, z]
    center_points = calculate_centroid(project_points)
    distance = max_euclidean_distance(center_points, project_points)
    x1 = xyzl[:, 0]
    y1 = xyzl[:, 1]
    z1 = xyzl[:, 2] - xyzl[:, 2]
    projecr_points2 = np.c_[x1, y1, z1]
    projecr_points2 = np.vstack([projecr_points2, center_points])
    project_cloud = o3d.geometry.PointCloud()
    project_cloud.points = o3d.utility.Vector3dVector(np.array(projecr_points2))
    pcd_terr = o3d.geometry.KDTreeFlann(project_cloud)
    [k1, idx1, _] = pcd_terr.search_radius_vector_3d(project_cloud.points[-1], distance - 5)
    points = xyzl[idx1[1:]]
    point1 = points[points[:, -2] == 1]
    point_o = points[np.argmin(points[:, 2])]
    xyz = xyzl[:, :3]
    xyzl = xyzl[xyzl[:, -2] != 1]
    point_l = np.argmax(xyzl[:, 2])
    point1 = np.argmax(xyzl[:, 1])
    point2 = np.argmax(xyzl[:, 0])
    point11 = np.argmin(xyzl[:, 1])
    point22 = np.argmin(xyzl[:, 0])
    h = xyzl[point_l][2] - point_o[2]
    w1 = math.sqrt((xyzl[point1][1] - xyzl[point11][1]) ** 2 + (xyzl[point1][0] - xyzl[point11][0]) ** 2)
    w2 = math.sqrt((xyzl[point2][0] - xyzl[point22][0]) ** 2 + (xyzl[point2][1] - xyzl[point22][1]) ** 2)
    if w1 > w2:
        return h,w1
    else:
        return h,w2

def max_euclidean_distance(base_point, points):
    '''

    :param base_point:Base point
    :param points:PointS in a point cloud
    :return:Mean distance
    '''
    base_point = np.array(base_point)
    other_points = np.array(points)

    # 计算差异向量
    differences = other_points - base_point

    # 计算每个差异向量的欧氏距离
    distances = np.sqrt(np.sum(differences ** 2, axis=1))

    # 计算平均距离
    avg_distance = np.mean(distances)

    return avg_distance

def calculate_centroid(points):
    """
    - points: An array of point clouds with the shape (N, 3), where N is the number of points.
    - centroid: Centroid coordinates.
    """
    centroid = np.mean(points, axis=0)
    return centroid

name=['ID']
leng=['株高']
width=['真实株幅']
data_root="F:\/240823_n40_data_train_val_test_result\普通手动真值"
for item in os.listdir(data_root):
    name.append(item)
    xyzl=np.loadtxt(os.path.join(data_root,item))
    h,w=P_HW(xyzl)
    print(h,w)
    width.append(w)
    leng.append(h)





import openpyxl

wb = openpyxl.Workbook()
sheet = wb.active
data=[name,leng,width]
id=1
for item in range(len(data)):
    index = id + item
    for i in range(len(data[item])):
        sheet.cell(row=i + 1, column=index, value=data[item][i])
print(name)
wb.save('株高株幅预测.xlsx')
