import numpy as np
import os

data1_root='F:\/240823_n40_data_train_val_test_result\inst_pred_best\/'
data2_root='F:\/240823_n40_data_train_val_test_result\sem_pred_best\/'
for item in os.listdir(data1_root):
    for item2 in os.listdir(data2_root):
        print(item2)
        if item!=item2:
            continue

        xyz1 = np.loadtxt(
            os.path.join(data1_root,item))
        xyz2 = np.loadtxt(
            os.path.join(data2_root,item2))
        print(np.unique(xyz1[:, -2]))
        print(np.unique(xyz2[:, -2]))
        data_l = xyz2[xyz2[:, -1] > 1]
        print(data_l.shape)
        print(xyz1.shape)
        data_p = xyz2[xyz2[:, -1] < 2]

        n_l = -1 * np.ones((data_l.shape[0], 1))
        n_p = np.ones((data_p.shape[0], 1))
        ll = np.unique(xyz1[:, -2])
        lp = np.unique(data_p[:, -1])
        print(ll, lp)

        for i in ll:
            if i > -1:
                i = i + 2
            n_l[xyz1[:, -2] == i] = i

        for y in lp:
            if y == 2:
                n_p[data_p[:, -1] == y] = 1
                continue
            n_p[data_p[:, -1] == y] = y
        print('1', np.unique(n_l))
        print('1', np.unique(n_p))
        ns=np.vstack([n_p,n_l])
        datas=np.vstack([data_p[:, :-1],data_l[:, :-1]])
        data1 = np.c_[data_l[:, :-1], n_l]
        data2 = np.c_[data_p[:, :-1], n_p]
        data = np.c_[datas, ns]
        # data = np.vstack([data1, data2])
        print(xyz2.shape)
        print(data.shape)
        np.savetxt('F:\/240823_n40_data_train_val_test_result\data\/'+item, data)
