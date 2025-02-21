import numpy as np
from numpy import linalg as LA
import os
def read_point_cloud(file_path):
    # 从TXT文件读取点云数据
    return np.loadtxt(file_path)

def project_to_xy(points):
    '''

    :param points: Plant point cloud data
    :return: Point cloud data projected onto the xy plane
    '''

    return points[:, :2]

def compute_centroid(points):
    '''

    :param points: Point cloud data projected onto the xy plane
    :return: Point set center point coordinates
    '''

    return np.mean(points, axis=0)

def covariance_matrix(points):
    '''

    :param points: Point cloud data projected onto the xy plane
    :return:Covariance matrix
    '''

    return np.cov(points, rowvar=False)

def main_principal_component(cov_matrix):
    '''

    :param cov_matrix: Covariance matrix
    :return: Principal component vector
    '''
    # 计算并返回主成分
    eigenvalues, eigenvectors = LA.eig(cov_matrix)
    return eigenvectors[:, np.argmax(eigenvalues)]

def project_points_on_line(points, line_point, line_direction):
    '''

    :param points: Point cloud data projected onto the xy plane
    :param line_point: Point set center point coordinates
    :param line_direction: Principal component vector
    :return: Terminal point
    '''
    # 投影点到直线
    line_point = np.array(line_point)
    line_direction = np.array(line_direction)
    points_from_origin = points - line_point
    projections = np.dot(points_from_origin, line_direction)[:, np.newaxis] * line_direction
    return line_point + projections

def max_euclidean_distance(points):
    '''

    :param points:Terminal point
    :return:Maximum distance
    '''
    # 计算最大欧氏距离
    return np.max(LA.norm(points - points[:, np.newaxis], axis=2))

# 使用示例

def PW(points):
    '''

    :param points: Plant point cloud data
    :return: Plant wight
    '''
    points = points[points[:, -1] > 0][:, :3]
    points_xy = project_to_xy(points)
    centroid = compute_centroid(points_xy)
    cov_matrix = covariance_matrix(points_xy)
    principal_component = main_principal_component(cov_matrix)
    projected_points = project_points_on_line(points_xy, centroid, principal_component)
    max_distance = max_euclidean_distance(projected_points)
    print("最大欧氏距离是:", max_distance)
    return max_distance


def main():
    data_root="D:\新花生\stem\/result\sem_pred"
    for item in os.listdir(data_root):
        # xyzl = np.loadtxt(os.path.join(data_root, item))
        points = read_point_cloud(os.path.join(data_root, item))
        PW(points)


if __name__=='__main__':
    main()