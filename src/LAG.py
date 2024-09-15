import numpy as np
from sklearn.decomposition import PCA
import os


def LAG(data):
    '''

    :param data:Leaf point cloud data
    :return:Blade Angle value
    '''
    xtzl=data
    xtzl = xtzl[xtzl[:, -2] == 2]
    l = np.unique(xtzl[:, -1])
    print(l)
    for i in l:
        points = xtzl[xtzl[:, -1] == i][:, :3]
        print(len(points))
        if len(points) < 4:
            continue
        pca = PCA(n_components=3)
        pca.fit(points)
        components = pca.components_  # 主成分向量
        third_component = components[2]  # 第三主成分
        print(third_component)
        # 计算第三主成分与Z轴的夹角
        z_axis = np.array([0, 0, 1])  # Z轴向量
        dot_product = np.dot(third_component, z_axis)
        # 确保第三主成分向量指向Z轴正方向
        if dot_product < 0:
            third_component = -third_component  # 翻转向量方向

        # 重新计算点积（如果已翻转，这一步是必要的）
        dot_product = np.dot(third_component, z_axis)
        norm_third = np.linalg.norm(third_component)
        norm_z = np.linalg.norm(z_axis)
        cos_angle = dot_product / (norm_third * norm_z)
        angle = np.arccos(cos_angle)  # 弧度
        angle_degrees = np.degrees(angle)  # 转换为度
        a.append(90 - angle_degrees)
    sun_a = sum(a)
    a1 = sun_a / len(a)
    return a1

name=['ID']
avg=['叶片角度']
data_root='F:\/240823_n40_data_train_val_test_result\普通手动真值'
for item in os.listdir(data_root):
    a=[]
    name.append(item)
    print(os.path.join(data_root,item))
    xtzl=np.loadtxt(os.path.join(data_root,item))
    a1=LAG(xtzl)
    avg.append(a1)

import openpyxl

wb = openpyxl.Workbook()
sheet = wb.active
data = [name, avg]
id = 1
for item in range(len(data)):
    index = id + item
    for i in range(len(data[item])):
        sheet.cell(row=i + 1, column=index, value=data[item][i])
wb.save('叶片角度预测.xlsx')




