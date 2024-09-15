import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Polygon
import open3d as o3d

import matplotlib.pyplot as plt

def PA(data):
    xyzl=data
    data = xyzl[xyzl[:, -2] == 2]
    max = np.max(data[:, -1])
    min = np.min(data[:, -1])
    l = np.unique(data[:, -1])
    lenth = []
    width = []
    idx = []
    for i in l:
        if i == -1:
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[data[:, -1] == float(i)][:, :3])

        X = np.asarray(pcd.points)
        # -----------------------------PCA------------------------------------
        pca = PCA(n_components=3)
        if X.shape[0] < 3:
            continue

        pca.fit(X)

        X_pca = pca.transform(X)
        project_points = X_pca
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(project_points)


        points = np.array(pcd.points)
        points_xy = points[:, :2]

        tri = Delaunay(points_xy)
        total_area = 0.0
        for simplex in tri.simplices:
            triangle = points_xy[simplex]
            polygon = Polygon(triangle)
            total_area += polygon.area
        total_area=total_area*100
        if total_area > 20 or total_area<2:
            print(i, total_area)
            continue
        area.append(total_area )

    aread = detect_outliers2(list(area[1:]))
    n = sum(aread)
    print(item, aread)
    # print('n',n)
    print('max', item, np.max(aread))
    print('min', item, np.min(aread))
    # return (n / len(aread))
    return area

def radius_outlier_removal(pcd, radius, min_neighbors):
    """
    Use radius outlier removal filter to denoise.
    :param pcd: Input the point cloud
    :param radius: Search radius
    :param min_neighbors: The minimum number of neighbors within the radius
    :return: Point cloud after de-noising
    """
    clean_pcd, ind = pcd.remove_radius_outlier(radius=radius, nb_points=min_neighbors)
    return clean_pcd

def detect_outliers2(df):
    outlier_indices = []

    Q1 = np.percentile(df, 25)

    Q3 = np.percentile(df, 75)

    IQR = Q3 - Q1

    # outlier step
    outlier_step = 1.5 * IQR

    i=0
    d=[]
    for nu in df:
        i=i+1
        if (nu > Q1 - outlier_step) and (nu < Q3 + outlier_step):
            d.append(nu)
        # print(i)

    return d

from sklearn.decomposition import PCA

num_b=1

import openpyxl
import os
data_root = 'D:\datains\论文data\分割结果\End2End白盆'
a = []
name = ['id']
for item in os.listdir(data_root):
    area = []
    area.append(item)
    xyzl = np.loadtxt(os.path.join(data_root,item))
    pa=PA(xyzl)
    a.append(pa)
    print(a)
import openpyxl
wb = openpyxl.Workbook()
sheet = wb.active


data=a

id = 1
for item in range(len(data)):
    index = id + item
    for i in range(len(data[item])):
        sheet.cell(row=i + 1, column=index, value=data[item][i])


wb.save('End2End叶片面积预测.xlsx')

