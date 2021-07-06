import numpy as np
import cv2
import os
def Bilateral_filter(depth_image,kernel):
    #产生卷积核参数
    img_x, img_y = depth_image.shape
    kernel_size = kernel
    center = kernel_size//2
    sigma = 2
    s = 2*(sigma**2)
    kernel, kernel_depth = np.zeros([kernel_size, kernel_size]), np.zeros([kernel_size, kernel_size])
    ############################
    addLine = int((kernel_size-1)/2)#由于kernel_size在对图像边缘进行滤波时候会超出图像范围，需要给周围添加0元素
    img = cv2.copyMakeBorder(depth_image, addLine, addLine, addLine, addLine, cv2.BORDER_REPLICATE)#扩充边缘
    source_x = addLine#从扩充后的索引开始
    source_y = addLine
    for delta_x in range(0,img_x):
        for delta_y in range(0,img_y):
            sum_val = 0
            #准备深度通道滤波和位置滤波
            for i in range(0, kernel_size):
                for j in range(0, kernel_size):
                    x = i-center
                    y = j-center
                    kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
                    kernel_depth[i, j] = np.exp(abs((int(img[source_x+i-center, source_y+j-center])-int(img[source_x, source_y]))) / s)
                    if img[source_x+i-center, source_y+j-center] == 0:
                        kernel_depth[i,j]=0
                    sum_val += kernel[i, j]*kernel_depth[i,j]
            if sum_val!=0:
                sum_val = 1 / sum_val#归一化
            kernel_mul = kernel*kernel_depth * sum_val
            ##结束
            if img[source_x,source_y]!=0:
                img[source_x, source_y] = np.sum(img[source_x-addLine:source_x+addLine+1,source_y-addLine:source_y+addLine+1]*kernel_mul)
            source_y = source_y+1
        source_x = source_x+1
        source_y = addLine
        new_image = img[addLine:img_x + addLine, addLine:img_y + addLine]
    return new_image
if __name__=="__main__":
    dir = "D:\CV_paper\depth\depth\depth_raw"
    ground_dir = "D:\CV_paper\depth\depth\depth_gt"
    depth_image =cv2.imread(os.path.join(dir,os.listdir(dir)[0]),0)
    ground_image =cv2.imread(os.path.join(ground_dir,os.listdir(dir)[0]),0)
    new_image = Bilateral_filter(depth_image, 3)
    cv2.imshow("new_image", new_image)
    cv2.imwrite("./image/bilateral_filter.jpg", new_image)
    cv2.waitKey()