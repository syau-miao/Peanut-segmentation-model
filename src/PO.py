import numpy as np
import os
import open3d as o3d
import math

def PO (data):
    '''

    :param data: Plant point cloud data.
    :return: Point cloud degree of spread.

    '''
    xyzl1=data
    label1 = np.unique(xyzl1[:, -1])

    points = xyzl1[xyzl1[:, -2] == 1]
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
    x1 = xyzl1[:, 0]
    y1 = xyzl1[:, 1]
    z1 = xyzl1[:, 2] - xyzl1[:, 2]
    projecr_points2 = np.c_[x1, y1, z1]
    projecr_points2 = np.vstack([projecr_points2, center_points])
    project_cloud = o3d.geometry.PointCloud()
    project_cloud.points = o3d.utility.Vector3dVector(np.array(projecr_points2))
    pcd_terr = o3d.geometry.KDTreeFlann(project_cloud)
    [k1, idx1, _] = pcd_terr.search_radius_vector_3d(project_cloud.points[-1], distance - 4)
    print(np.max(np.array(idx1)))
    print(project_cloud)
    points = xyzl1[idx1[1:]]
    point11 = points[points[:, -1] == 1]
    print(len(point11))
    point_o = points[np.argmin(points[:, 2])]
    point1 = point_o[:3] + [0, 0, 10]
    distance2 = max_euclidean_distance(point_o[:3], xyzl1[xyzl1[:, -1] > 0][:, :3])

    for i in xyzl1[xyzl1[:, -2] !=1 ]:

        d = np.linalg.norm(i[:3] - point_o[:3])
        if d < distance2:
            continue
        arg = cal_angle(i[:3], point_o[:3], point1)
        if i[2] <= point_o[2]:
            continue
        if i[1] > point_o[1]:
            ar.append(arg)
        else:
            ar2.append(arg)
    print(item)
    print(len(ar), len(ar2))
    if len(ar) == 0:
        m = 0 + max(ar2)
    elif len(ar2) == 0:
        m = 0 + max(ar)
    else:

        m = max(ar) + max(ar2)
    return m

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

    return B
def calculate_centroid(points):
    """
    - points: An array of point clouds with the shape (N, 3), where N is the number of points.
    - centroid: Centroid coordinates.
    """
    centroid = np.mean(points, axis=0)
    return centroid
def last_length(data,point):
    min_dis=10000
    p=np.zeros((data.shape[0],data.shape[1]))
    for i in data:
        dis=np.sqrt(np.sum((i[:3] - point) ** 2))
        if dis<min_dis:
            min_dis=dis
            p=i
    return p

def max_euclidean_distance(base_point, points):
    '''

    :param base_point:Base point
    :param points:PointS in a point cloud
    :return:Mean distance
    '''
    base_point = np.array(base_point[:3])
    other_points = np.array(points[:,:3])

    differences = other_points - base_point


    distances = np.sqrt(np.sum(differences ** 2, axis=1))


    avg_distance = np.mean(distances)

    return avg_distance

a=['PO']
name=['id']

# dataj_root="D:\py\j\/result"
data_root="F:\/240823_n40_data_train_val_test_result\普通手动真值"
for item in os.listdir(data_root):
    ar=[]
    ar2=[]
    name.append(item)
    max1 = 10000
    arg=0
    xyzl1=np.loadtxt(os.path.join(data_root,item))
    m=PO(xyzl1)
    a.append(m/180)
    print(a)

import openpyxl
wb = openpyxl.Workbook()
sheet = wb.active
sheet.title = 'Z3'
data=[name,a]
id=1
for item in range(len(data)):
    index = id + item
    for i in range(len(data[item])):
        sheet.cell(row=i + 1, column=index, value=data[item][i])
wb.save('扩展度预测.xlsx')



