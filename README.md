# 3D_tutorial
it's a simple 3D point cloud tutorial  

代码环境：WIN10  
PYTHON:Python 3.8.8  
cv2:4.5.2  
numpy:1.20.1  
open3d:0.12.0  
***  
程序功能：  
*pca_normal.py*  
1.能对给定地址的三维点云图像做PCA降维（将3维降到1维）  
2.求每个点的法向量并显示出来  
***  
运行  
python pca_normal.py --isPCA  1
对点云进行可视化  
python pca_normal.py --isPCA 2 (或者其他非1的int类型)  
对点云进行法向量的可视化
***  
*filter.py*  
对点云进行降采样  
python filter.py  
*Bilateral_new_filter.py*
对深度图进行双边滤波  
python Bilateral_new_filter.py  
效果图：  


