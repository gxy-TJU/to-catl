from PIL import Image
import depth_pro
import cv2
import numpy as np
import os
from torchvision.utils import save_image
# Load model and preprocessing transform
import torchvision.transforms as transforms
model, transform = depth_pro.create_model_and_transforms()
model.eval()

img_folder = '/home/gxy/work/manipulator/ml-depth-pro/test_data'
img_list = [os.path.join(img_folder,nm[:-4]) for nm in os.listdir(img_folder) if nm[-3:] in ['jpg']]
for i in range(1,100):
    I = Image.open(f'{img_list[i]}norm.png')
    oral_pic = np.array(I)/255*400
    image, _, f_px = depth_pro.load_rgb(f'{img_list[i]}.jpg')
    image = transform(image)
# Run inference.
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]   # Depth in [cm].
    depth_array = depth.cpu().numpy()
    #cv2.imwrite(f'{img_list[i]}_error', error)
# 2. 归一化到 [0, 255] 范围
    #depth_normalized = (255 * (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())).astype(np.uint8)
    depth_normalized = (255 * depth_array / 4)
    error = depth_array * 100 - oral_pic
    #error = np.rint(error)
    np.savetxt(f'{img_list[i]}_error', error.astype(int))
# 3. 保存为灰度图像
    cv2.imwrite(f"{img_list[i]}_predicted.png", depth_normalized)




#focallength_px = prediction["focallength_px"]  # Focal length in pixels.
# print(focallength_px)