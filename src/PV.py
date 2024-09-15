import open3d as o3d
import numpy as np
import openpyxl
import os

def volume_Voxel(data,step=0.2):
    '''

    :param data: Plant point cloud data
    :param step: Voxel side length
    :return: Point cloud volume
    '''
    pcd_cloud=data[:,:3]
    x_min, y_min, z_min = np.amin(pcd_cloud, axis=0)
    x_max, y_max, z_max = np.amax(pcd_cloud, axis=0)
    step = step
    along = int(np.ceil((x_max - x_min) / step))
    width = int(np.ceil((y_max - y_min) / step))
    hight = int(np.ceil((z_max - z_min) / step))
    m = np.zeros((int(along), int(width), int(hight)))
    for i in range(len(pcd_cloud)):
        x = np.floor((pcd_cloud[i][0] - x_min) / step)
        y = np.floor((pcd_cloud[i][1] - y_min) / step)
        z = np.floor((pcd_cloud[i][2] - z_min) / step)
        m[int(x), int(y), int(z)] = 1

    ind = 0
    for i in range(along):
        for j in range(width):
            for a in range(hight):
                if m[i, j, a] == 1:
                    ind += 1

    print('ind', ind)
    V = ind * 1.0 * 1 * 1
    return V



def main():
    data_root = "F:\/240823_n40_data_train_val_test_result\普通手动真值"
    v=['体积']
    name=['ID']
    for item in os.listdir(data_root):
        xyzl=np.loadtxt(os.path.join(data_root,item))
        xyzl=xyzl[xyzl[:,-2]!=1]
        name.append(item)
        points = []
        pcd=o3d.geometry.PointCloud()
        pcd.points=o3d.utility.Vector3dVector(xyzl[:,:3])
        print(pcd)
        step = 1
        pcd_cloud = np.array(pcd.points)
        V=volume_Voxel(pcd_cloud,step)
        print(V)
        v.append(V)
    wb = openpyxl.Workbook()
    sheet = wb.active
    data = [name, v]
    id = 1
    for item in range(len(data)):
        index = id + item
        for i in range(len(data[item])):
            sheet.cell(row=i + 1, column=index, value=data[item][i])
    wb.save('体积预测.xlsx')

if __name__=='__main__':
    main()



