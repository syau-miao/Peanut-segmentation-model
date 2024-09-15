import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.tri as tri
import tkinter as tk
import openpyxl
from tkinter import filedialog



def PA(data,step=0.10):
    '''

    :param data: Plant point cloud data
    :param step: Grid step size网格步长（默认为1.0）
    :return: Point cloud projection area
    '''

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])

    points = np.asarray(pcd.points)

    xi = points[:, 0]
    yi = points[:, 1]
    zi = points[:, 2] - points[:, 2]
    project_points = np.c_[xi, yi, zi]


    x_min, y_min, z_min = np.amin(project_points, axis=0)
    x_max, y_max, z_max = np.amax(project_points, axis=0)

    width = int(np.ceil((x_max - x_min) / step))
    height = int(np.ceil((y_max - y_min) / step))

    M = np.zeros((int(width), int(height)))

    for i in range(len(project_points)):
        row = np.floor((project_points[i][0] - x_min) / step)
        col = np.floor((project_points[i][1] - y_min) / step)
        M[int(row), int(col)] += 1


    ind = 0
    for i in range(width):
        for j in range(height):
            if M[i, j] > 0:
                ind = ind + 1

    area = ind * 1.0 * 1.0
    print(area)
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(project_points)

    return area


def main():
    data_root="F:\/240823_n40_data_train_val_test_result\普通手动真值"
    a=['投影面积']
    name=['ID']
    for item in os.listdir(data_root):
        xyzl=np.loadtxt(os.path.join(data_root,item))
        xyzl=xyzl[xyzl[:,-2]!=1]
        area = PA(xyzl, 1)
        name.append(item)
        a.append(area)
    wb = openpyxl.Workbook()
    sheet = wb.active
    data = [name, a]
    id = 1
    for item in range(len(data)):
        index = id + item
        for i in range(len(data[item])):
            sheet.cell(row=i + 1, column=index, value=data[item][i])

    wb.save('投影面积预测.xlsx')




if __name__=='__main__':
    main()