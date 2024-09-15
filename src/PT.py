import open3d as o3d
import numpy as np
import open3d.utility
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
import os
import math
import matplotlib.pyplot as plt
import copy
from sklearn.cluster import KMeans


sumage=0


def max_euclidean_distance(base_point, points):
    '''

    :param base_point:Base point
    :param points:PointS in a point cloud
    :return:Mean distance
    '''

    base_point = np.array(base_point)
    other_points = np.array(points)

    differences = other_points - base_point

    distances = np.sqrt(np.sum(differences ** 2, axis=1))


    avg_distance = np.mean(distances)

    return avg_distance

def cal_angle(point_a, point_b, point_c):

    """
    Calculate the Angle according to the three-point coordinates

                  a
           b ∠
                   c

    :param point_a、point_b、point_c: 3D coordinate
    :return: Returns the value of the Angle between corner b
    """
    a_x, b_x, c_x = point_a[0], point_b[0], point_c[0]
    a_y, b_y, c_y = point_a[1], point_b[1], point_c[1]

    if len(point_a) == len(point_b) == len(point_c) == 3:

        a_z, b_z, c_z = point_a[2], point_b[2], point_c[2]
    else:
        a_z, b_z, c_z = 0,0,0



    x1,y1,z1 = (a_x-b_x),(a_y-b_y),(a_z-b_z)
    x2,y2,z2 = (c_x-b_x),(c_y-b_y),(c_z-b_z)

    cos_b = (x1*x2 + y1*y2 + z1*z2) / (math.sqrt(x1**2 + y1**2 + z1**2) *(math.sqrt(x2**2 + y2**2 + z2**2))) # 角点b的夹角余弦值
    B = math.degrees(math.acos(cos_b))
    if c_z==point_b[-1]:
        if c_y<point_b[1]:
            B=360-B

    return B

def calculate_centroid(points):
    """
    - points: An array of point clouds with the shape (N, 3), where N is the number of points.
    - centroid: Centroid coordinates.
    """
    centroid = np.mean(points, axis=0)
    return centroid

def PT(data_root,item,data,bandwidth=0):
    '''

    :param data:Plant point cloud data
    :param bandwidth:Clustering step size
    :return:opacity（float）
    '''


    xyzl = np.loadtxt(os.path.join(data_root, item))

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
    [k1, idx1, _] = pcd_terr.search_radius_vector_3d(project_cloud.points[-1], distance - 4)
    points = xyzl[idx1[1:]]
    point11 = points[points[:, -1] == 1]
    print(len(point11))
    point_o = points[np.argmin(points[:, 2])]

    fig3 = plt.figure(num=3, figsize=(5, 5))
    axes3 = fig3.add_subplot(1, 1, 1)
    plt.xlim(0, 360)
    plt.ylim(0, 90)
    pcd_n = o3d.geometry.PointCloud()
    pcd_n.points = o3d.utility.Vector3dVector(data[:,:3])

    k = 20
    kmeans = KMeans(n_clusters=k,n_init=10)
    labels = kmeans.fit_predict(np.asarray(pcd_n.points))
    colors = plt.get_cmap("tab10")(labels / k)
    pcd_n.colors = o3d.utility.Vector3dVector(colors[:, :3])

    data=np.c_[data[:,:3],labels]

    sumage=0
    segment = []
    center = []
    sh_center = []
    obbpoint = []
    agez = []
    agey = []
    p_a = 0
    p_s = 0
    angle_s = []


    _, _, zmin = np.amin(pcd_n.points, axis=0)
    lpoint=point_o[:3]

    srclinex = lpoint + [0.2, 0, 0]
    srclinez = lpoint + [0, 0, 0.2]

    l=np.unique(data[:,-1])


    for i in l:
        clusters_cloud = o3d.geometry.PointCloud()
        clusters_cloud.points=o3d.utility.Vector3dVector(data[data[:,-1]==i][:,:3])

        segment.append(clusters_cloud)

        if len(np.array(clusters_cloud.points)) < 4:
            continue
        x_min, y_min, z_min = np.amin(clusters_cloud.points, axis=0)
        if z_min == lpoint[-1]:
            continue
        min_bound = clusters_cloud.get_min_bound()
        x_max, y_max, z_max = np.amax(clusters_cloud.points, axis=0)
        for i in [x_min, x_max]:
            for x in [y_min, y_max]:
                for y in [z_min, z_max]:
                    obbpoint.append([i, x, y])
        for p in obbpoint:
            p = np.array(p)
            a = copy.deepcopy(p)
            a[-1] = lpoint[-1]
            age_z = (cal_angle(srclinez, lpoint, p))
            age_y = cal_angle(srclinex, lpoint, a)
            agey.append(age_y)
            agez.append(age_z)
        max_y = max(agey)
        min_y = min(agey)
        max_z = max(agez)
        min_z = min(agez)
        if max_y > 270 and min_y > 0 and min_y < 90:
            y = 360 - max_y
            max_y = min_y + y
        p = plt.Polygon(xy=[[min_y, max_z], [max_y, max_z], [max_y, min_z], [min_y, min_z]], color='green', alpha=0.8)
        axes3.add_patch(p)
        age = (math.radians(max_y) - math.radians(min_y)) * (-math.cos(np.radians(max_z)) + math.cos(np.radians(min_z)))
        sumage += age

        if len(angle_s) != 0:
            p_max = []
            p=[]
            for i in range(len(angle_s)):

                xy_a = []
                xz_a = []
                a_dt1 = angle_s[i][0] - angle_s[i][1]
                a_dt2 = max_y - min_y
                s_dt1 = angle_s[i][2] - angle_s[i][3]
                s_dt2 = max_z - min_z
                a_angle = angle_s[i][1] - min_y
                s_angle = angle_s[i][3] - min_z
                if (a_angle - a_dt2) < 0 and (min_y - angle_s[i][1] - a_dt1) < 0 and (s_angle - s_dt2) < 0 and (
                        min_z - angle_s[i][3] - s_dt1) < 0:
                    xy_a = [angle_s[i][0], angle_s[i][1], max_y, min_y]
                    xz_a = [angle_s[i][2], angle_s[i][3], max_z, min_z]
                    xy_a.sort()
                    xz_a.sort()
                    p_a = (math.radians(xy_a[2]) - math.radians(xy_a[1])) * (
                                -math.cos(np.radians(xz_a[2])) + math.cos(np.radians(xz_a[1])))
                    p.append([xy_a[2],xy_a[1],xz_a[2],xz_a[1]])
                    p_s = p_s + p_a


                    p_max.append(p_a)


            if len(p)>1:
                p_a2 = 0
                for i in range(len(p)-1):

                    xy_a = []
                    xz_a = []
                    a_dt1 = p[i][0] - p[i][1]
                    a_dt2 = p[-1][0] - p[-1][1]
                    s_dt1 = p[i][2] - p[i][3]
                    s_dt2 = p[-1][2] - p[-1][3]
                    a_angle = p[i][1] - p[-1][1]
                    s_angle = p[i][3] - p[-1][3]
                    if (a_angle - a_dt2) < 0 and (p[-1][1] - p[i][1] - a_dt1) < 0 and (s_angle - s_dt2) < 0 and (
                            p[-1][3] - p[i][3] - s_dt1) < 0:
                        xy_a = [p[i][0], p[i][1], p[-1][0], p[-1][1]]
                        xz_a = [p[i][2], p[i][3], p[-1][2], p[-1][3]]
                        xy_a.sort()
                        xz_a.sort()
                        p_a = (math.radians(xy_a[2]) - math.radians(xy_a[1])) * (
                                -math.cos(np.radians(xz_a[2])) + math.cos(np.radians(xz_a[1])))
                        p_a2=p_a2+p_a
                p_s=p_s-p_a2


        angle_s.append([max_y, min_y, max_z, min_z])
        p_center = clusters_cloud.get_center()
        sp_center = p_center - [0, 0, p_center[-1] - z_min]
        center.append(p_center)
        sh_center.append(sp_center)
        obb = clusters_cloud.get_oriented_bounding_box()
        obb.color = (1, 0, 0)
        r_color = np.random.uniform(0, 1, (1, 3))
        clusters_cloud.paint_uniform_color([r_color[:, 0], r_color[:, 1], r_color[:, 2]])
        segment.append(clusters_cloud)
        segment.append(obb)
        obbpoint = []
        agey = []
        agez = []

    s = sumage - p_s
    a = 1-(s / 6.28)

    if a<0:
        print(s)
    print(a)
    return a

import tkinter as tk

from tkinter import filedialog
if __name__=='__main__':
    import openpyxl

    data_root = 'F:\/240823_n40_data_train_val_test_result\普通手动真值'
    a = ['透光率']
    name = ['id']
    for item in os.listdir(data_root):
        xyzl = np.loadtxt(os.path.join(data_root, item))
        xyzl = xyzl[xyzl[:, -2] > 1]
        p_n = xyzl
        name.append(item)
        area=PT('F:\/240823_n40_data_train_val_test_result\普通手动真值',item,p_n)
        a.append(area)


    wb = openpyxl.Workbook()
    sheet = wb.active
    data = [name, a]
    id = 1
    for item in range(len(data)):
        index = id + item
        for i in range(len(data[item])):
            sheet.cell(row=i + 1, column=index, value=data[item][i])
    wb.save('透光率.xlsx')


