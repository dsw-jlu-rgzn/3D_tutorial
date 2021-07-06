import numpy as np
from math import floor
import open3d as o3d
from pca_normal import get_matrix
def voxel_filter(Sample_matrix, leaf=10):
    #求所有点云的最大最小值以便确定过滤范围
    min = Sample_matrix.min(0)
    max = Sample_matrix.max(0)
    #过滤的最小单位
    bin = (max - min) / leaf
    len_sample, _ = Sample_matrix.shape
    the_lo = np.zeros((10000, 1))
    for i in range(len_sample):
        #对于每一个点，求出这个点在网格的哪一个位置
        point = Sample_matrix[i]
        the_lo[i] = floor((point[0] - min[0]) / bin[0]) + floor((point[1] - min[1]) / bin[1] * leaf) + floor(
            (point[2] - min[2]) / bin[2] * leaf * leaf)
    #对点云按照位置进行排序
    index_sort = np.argsort(the_lo.T)
    new_point = []
    temp_point = []
    #点云的过滤，对于每一个网格来说，随机取一个点
    for i in range(len(index_sort[0])):
        if len(temp_point) == 0:
            temp_point.append(the_lo[index_sort[0, i]])
        else:
            if temp_point[0] != the_lo[index_sort[0, i]]:  # 这个元素与之前容器里不相等，则换容器
                ran = np.random.randint(0, len(temp_point))  # 随机取其中元素
                new_point.append(temp_point[ran])
                temp_point.clear()  # 将容器里面元素清除
                temp_point.append(the_lo[index_sort[0, i]])
            else:  # 当前元素与容器里面元素相同
                temp_point.append(the_lo[index_sort[0, i]])
    new_point = np.array(new_point)
    the_reshape = new_point.reshape((-1,)).astype(int)
    the_matrix = Sample_matrix[the_reshape]
    return the_matrix
if __name__=="__main__":
    #dir = "D:/CV_paper/shenlanxueyuan/modelnet40_normal_resampled/modelnet40_normal_resampled/airplane/airplane_0001.txt"
    All_matrix = []
    with open("dir_object.txt", 'r') as f:
        file_ = f.readlines()
        for i in range(len(file_)):
            file_[i] = file_[i].strip('\n')
            matrix = get_matrix(file_[i])
            All_matrix.append(matrix)
    Sample_matrix = All_matrix[0][:, :3]#拿到样本点云（飞机）
    the_matrix = voxel_filter(Sample_matrix,leaf=15)#进行voxel过滤
    #点云可视化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(the_matrix)

    o3d.visualization.draw_geometries([pcd])






