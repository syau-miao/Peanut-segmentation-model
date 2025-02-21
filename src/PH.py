import math

import numpy as np
import os
import math
import open3d as o3d
import openpyxl

def mean_absolute_percentage_error(y_true, y_pred):
    '''

    :param y_true: True value
    :param y_pred: Predicted value
    :return: MAPE
    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    y_true = np.where(y_true == 0, np.finfo(float).eps, y_true)  # Replace 0 with a very small number
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    return mape

def P_HW(data,points,num):
    '''

    :param data: Plant point cloud data
    :param points: Pot point cloud data
    :param num: The column where the label is located
    :return:Plant height
    '''
    xyzl=data
    print(xyzl)
    points = points
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
    point1 = points[points[:, num] !=0]
    point_o = point1[np.argmin(point1[:, 2])]
    xyzl = xyzl[xyzl[:, num] != 0]
    point_l = np.argmax(xyzl[:, 2])
    h = xyzl[point_l][2] - point_o[2]
    return h


def max_euclidean_distance(base_point, points):
    '''

    :param base_point:Base point
    :param points:PointS in a point cloud
    :return:Mean distance
    '''
    # 将列表转换为NumPy数组以进行向量化操作
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
    # 计算质心
    centroid = np.mean(points, axis=0)
    return centroid

def main():
    name = ['ID']
    leng = ['株高']
    leng_pred = ['pred_high']
    data_root = "D:\新花生\stem\/result\sem_pred"
    for item in os.listdir(data_root):
        print(item)
        name.append(item)
        xyzl = np.loadtxt(os.path.join(data_root, item))
        points = xyzl[xyzl[:, -1] == 0]
        h = P_HW(xyzl, points, -1)

        leng.append(h)
        points_pred = xyzl[xyzl[:, -2] == 0]
        h_pred = P_HW(xyzl, points_pred, -2)
        # print(h, w)
        leng_pred.append(h_pred)



    wb = openpyxl.Workbook()
    sheet = wb.active
    name.append("")
    mape_h = mean_absolute_percentage_error(leng[1:], leng_pred[1:])

    leng.append("MAPE")
    leng_pred.append(mape_h)

    data = [name, leng, leng_pred]
    id = 1
    for item in range(len(data)):
        index = id + item
        for i in range(len(data[item])):
            sheet.cell(row=i + 1, column=index, value=data[item][i])
    print(name)
    wb.save('D:\py\表型\RESULT\株高预测.xlsx')

if __name__=='__main__':
    main()
