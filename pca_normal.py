import numpy as np
import open3d as o3d
import os
import argparse
def get_matrix(dir):
    '''
    :param dir: txt文件位置
    :return: 点云矩阵
    '''
    with open(dir, "r") as f:
        data = f.read()
        data_list = data.strip().split('\n')
        for i in range(len(data_list) ):
            data_list[i] = data_list[i].split(',')
            data_list[i] = [float(j) for j in data_list[i]]
    data_matrix = np.array(data_list)
    return data_matrix
def PCA_new(data_matrix,correlation=False,sort=True):
    """
    :param data_matrix: sample_number*3
    :param correlation: True/False
    :param sort: when True,DownDimension;False:normal vector
    :return:  sample_number* (reduction dimension)
    """
    data_mean = np.mean(data_matrix, axis=0)#求均值
    shape_m, shape_n = data_matrix.shape
    average = np.tile(data_mean, (shape_m, 1))
    data_adj = data_matrix - average#把数据归一化
    if correlation:#这个不知道能干啥...先写着
        pass
    cov = np.dot(data_adj.T, data_adj)
    eig_values, eig_vector = np.linalg.eig(cov)
    #print(eig_values)
    indexs_ = np.argsort(-eig_values)
    if sort:#排序的时候PCA用于降维
        picked_eig_values = eig_values[indexs_]
        picked_eig_vector = eig_vector[:, indexs_]
        data_ndim = np.dot(data_matrix, picked_eig_vector)
    else:
        indexs_ = np.argsort(eig_values)
        picked_eig_values = eig_values[indexs_]
        picked_eig_vector = eig_vector[:, indexs_]
    return  picked_eig_values,picked_eig_vector
def get_normal_vector(matrix):
    All_normal_vactor = []
    shape_m, shape_n = matrix.shape
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(matrix)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for i in range(shape_m):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], 0.05)
        if k>0:
            pre_matrix = matrix[idx, :]
            _, normal_vector = PCA_new(pre_matrix, sort=False)
            normal_vector = normal_vector[:, 0]
            All_normal_vactor.append(normal_vector)
    the_normal_vector = np.array(All_normal_vactor,dtype=np.float64)
    return  the_normal_vector
def get_dir_txt(dir):
    all_dir = []
    with open("dir_object.txt", 'w') as f:
        for i in os.listdir(dir):
            temp_dir = os.path.join(dir, i)
            if os.path.isdir(temp_dir):
                file_name = os.listdir(temp_dir)[0]
                file_path = os.path.join(temp_dir, file_name)
                file_path = file_path.replace('\\', '/')
                print(file_path)
                f.writelines(file_path + '\n')
                all_dir.append(file_path)
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--isPCA', default=1, type=int)#默认为进行PCA降维可视化，否则进行法向量可视化(其他数字)
    args = parser.parse_args()
    #dir是点云的位置
    dir = "D:/CV_paper/shenlanxueyuan/modelnet40_normal_resampled/modelnet40_normal_resampled"
    #生成样本点云地址的txt文件
    get_dir_txt(dir)
    All_matrix = []
    #将所有样本点云转成matrix格式 number*6
    with open("dir_object.txt", 'r') as f:
        file_ = f.readlines()
        for i in range(len(file_)):
            file_[i] = file_[i].strip('\n')
            matrix = get_matrix(file_[i])
            All_matrix.append(matrix)
    #取第一个点云（飞机）
    Sample_matrix = All_matrix[0][:, :3]
    #生成点云的法向量
    the_normal_vector = get_normal_vector(Sample_matrix)
    #用PCA分析点云主方向
    Sample_PCA = np.dot(Sample_matrix,PCA_new(Sample_matrix)[1][:,:2])
    shape_S,_ = Sample_matrix.shape
    z_new = np.zeros((shape_S,1))
    Sample_PCA = np.c_[Sample_PCA,z_new]
    pcd = o3d.geometry.PointCloud()
    is_PCA = args.isPCA
    #用args.isPCA决定可视化主方向还是法向量
    if is_PCA==1:
        pcd.points = o3d.utility.Vector3dVector(Sample_PCA)
        o3d.visualization.draw_geometries([pcd])
    else:
        pcd.points = o3d.utility.Vector3dVector(Sample_matrix)
        pcd.normals = o3d.utility.Vector3dVector(the_normal_vector)
        o3d.visualization.draw_geometries([pcd])
