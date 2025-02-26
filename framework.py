
from openai import OpenAI
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
from pydantic import BaseModel
from ollama import chat
from PIL import ImageDraw
import numpy as np
import cv2


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "IDEA-Research/grounding-dino-base"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

'''用户指令解析'''
def instruction2json(user_instruction = 'move the alarm clock to the left of the laptop'):
    class FriendInfo(BaseModel):
        pick: str
        place: str
        relationship: str

    system_prompt = f'''
    Please convert the following user instruction into a JSON object. 
    Identify the object to pick, the reference object for placement, 
    and the spatial relationship between them.
    The spatial relationship can be 'left', 'right', 'on'.
    '''
  
    response = chat(
        'deepseek-r1:1.5b',
        messages=[
            {'role': 'system', 'content': f'{system_prompt}'},
            {'role': 'user', 'content': 'give the apple to the panda'},
            {'role': 'assistant', 'content': ' "pick": "fish", "place": "panda", "relationship": "left" '},
            {'role': 'user', 'content': "place the fish to the left of the panda"},
            {'role': 'assistant', 'content': ' "pick": "fish", "place": "panda", "relationship": "left" '},
            {'role': 'user', 'content': f"{user_instruction}"},
        ],
        format=FriendInfo.model_json_schema(),  # Use Pydantic to generate the schema or format=schema
        options={'temperature': 0},
    )
    
    friends_response = FriendInfo.model_validate_json(response.message.content)
    print(friends_response.pick, friends_response.place, friends_response.relationship)
    return friends_response.pick, friends_response.place, friends_response.relationship

def text2box(text, image = 'c:\\Users\\Administrator\\Desktop\\R-C.jpg'):
    image = Image.open(image)
    print(text)
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    print(results)
    boxes = results[0]['boxes'].cpu().numpy()
    return boxes

def draw_boxes_on_frame(frame, pick_object, place_object, pick_result, place_result):
    image = Image.open(frame)
    draw = ImageDraw.Draw(image)
    
    pick_corrd = pick_result[0] if len(pick_result) > 0 else None
    place_corrd = place_result[0] if len(place_result) > 0 else None
    
    if pick_corrd is not None:
        draw.rectangle(pick_corrd, outline="red", width=3)
        draw.text((pick_corrd[0], pick_corrd[1]), pick_object, fill="red")
    
    if place_corrd is not None:
        draw.rectangle(place_corrd, outline="blue", width=3)
        draw.text((place_corrd[0], place_corrd[1]), place_object, fill="blue")
    
    image.show()

def get_corrd_from_instruction(user_instruction, frame):
    pick_object, place_object, relation = instruction2json(user_instruction=user_instruction)

    pick_result = text2box(pick_object, frame)
    place_result = text2box(place_object, frame)
    print(pick_result[0], place_result)
    draw_boxes_on_frame(frame, pick_object, place_object, pick_result, place_result)
    
    pick_corrd = np.array([[(pick_result[0][0] + pick_result[0][2])/2, (pick_result[0][1] + pick_result[0][3])/2]]) if len(pick_result) > 0 else None
    place_corrd = np.array([[(place_result[0][0] + place_result[0][2])/2, (place_result[0][1] + place_result[0][3])/2]]) if len(place_result) > 0 else None
    
    print(pick_corrd, place_corrd, relation)
    return pick_corrd, place_corrd, relation


'''pixel2world'''

def camera_params(frame):
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

    return K, D, R, T, Z
def pixel2world(coord):
    K, D, R, T, Z = camera_params()
    undistorted_pixel_coords = cv2.undistortPoints(coord, cameraMatrix =  K, distCoeffs = D, P = K)
    image_point_cam = np.array([undistorted_pixel_coords[0][0][0],
                               undistorted_pixel_coords [0][0][1], 1.0])
    image_point_cam = image_point_cam.reshape((3, 1))
    undistorted_pixel_coords_homogeneous = image_point_cam * Z 
    K_inv = np.linalg.inv(K)
    camera_coords = K_inv.dot(undistorted_pixel_coords_homogeneous)
    T1 = T.reshape((3, 1))
    Rt = np.hstack((R, T1))
    #print('Rt', Rt)
    Rt = np.vstack((Rt, [0, 0, 0, 1]))
    # 计算组合矩阵的逆
    Rt_inv = np.linalg.inv(Rt)
    world_points = Rt_inv.dot(np.append(camera_coords, 1))
    x = world_points[0]
    y = world_points[1]
    return

'''执行抓取操作'''
def pick(coord):
    return


'''执行放置操作'''
def place(corrd):

    return 

if __name__ == '__main__':
    get_corrd_from_instruction('move the clock to left of the time', 'c:\\Users\\Administrator\\Desktop\\R-C.jpg')
    pass