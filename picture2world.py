import numpy as np
import cv2

#一些已知的矩阵
#相机内参矩阵
K = np.array([[131.0976, 0,  60.5716],
                [0, 131.0591, 57.9064],
                [0,  0,  1]], dtype=np.float32)
# 示例相机畸变系数
D = np.array([-0.4243, 0.2247,  -0.00054117, -0.00029614, -0.0649,], dtype=np.float32)
#相机旋转矩阵
R = np.array([[ 0.9996, -0.0006,  0.0271],
                [0.0001, 0.9998, 0.0197],
                [-0.0271,  -0.0197, 0.9994]], dtype=np.float32)
# 相机的平移矩阵
T = np.array([[-31.0434],
            [ 47.5918],
            [ 329.6273]], dtype=np.float32)  
# 假设物体在相机坐标系中的深度（Z）
     # 需要已知或通过其他方式获取深度值
Z = 335
def image_to_world(x, y):

# 像素坐标（pixel_coords）
    pixel_coords = np.array([[x, y]], dtype=np.float32)


    #print(pixel_coords)
# 将像素坐标进行畸变矫正
    undistorted_pixel_coords = cv2.undistortPoints(pixel_coords, cameraMatrix =  K, distCoeffs = D, P = K)
    #undistorted_pixel_coords = undistorted_normalized_coords[0][0] * np.array([fx, fy]) + np.array([cx, cy])
    #print(undistorted_pixel_coords)
# 将像素坐标转换为齐次坐标
    image_point_cam = np.array([undistorted_pixel_coords[0][0][0],
                               undistorted_pixel_coords [0][0][1], 1.0])
    image_point_cam = image_point_cam.reshape((3, 1))
    #print(image_point_cam)
    # 计算相机坐标系中的点
    undistorted_pixel_coords_homogeneous = image_point_cam * Z 
    K_inv = np.linalg.inv(K)
    camera_coords = K_inv.dot(undistorted_pixel_coords_homogeneous)
    #print(camera_coords)

    T1 = T.reshape((3, 1))
    #print('R', R, 'T', T1)

    # 组合旋转矩阵和平移向量，形成一个4x4矩阵
    Rt = np.hstack((R, T1))
    #print('Rt', Rt)
    Rt = np.vstack((Rt, [0, 0, 0, 1]))

    # 计算组合矩阵的逆
    Rt_inv = np.linalg.inv(Rt)
    world_points = Rt_inv.dot(np.append(camera_coords, 1))

    # 计算世界坐标
    #world_coords_homogeneous = Rt_inv.dot(np.append(camera_coords, 1))
    #world_coords = world_coords_homogeneous[:3]
    #print(world_points[:2])
    x = world_points[0]
    y = world_points[1]
    #print(x, y)
    #print('x', x, 'y' , y)
    #122
    cy = 37.5 + x
    #235
    cx = 153 + y
    return cx, cy
    

if __name__ =="__main__" :
    # 图像上的点坐标
    # 计算对应的世界坐标
    x = 48
    y = 84
    cx, cy = image_to_world(x, y)
    cx = int(cx)
    cy = int(cy)
    print(cx, cy)