# -*- coding:utf-8 -*-
import sys
import face_recognition
import cv2
import os
from PIL import Image

# print("检测目标文件路径："+sys.argv[1])
location_1 = sys.argv[1]
# 正常文件结果集
normal_results = []
normal_results.append("normal\image0 name:zhangbozhi")
normal_results.append("normal\image1 name:liudehua")
normal_results.append("normal\image2 name:liuyifei")
# 警告文件结果集
warning_results = []
warning_results.append("warning\image0 name:mayun")

im = Image.open(location_1)

size = im.size
if size[0] > 1000:
    if size[0] > size[1]:
        rate = float(1000) / float(size[0])
    else:
        rate = float(750) / float(size[1])
    new_size = (int(size[0] * rate), int(size[1] * rate))
    new = im.resize(new_size, Image.BILINEAR)
    new.save(location_1)

# 正常文件结果集
normal_results = []
normal_results.append("name:zhangbozhi")
normal_results.append("name:liudehua")
normal_results.append("name:liuyifei")

# 警告文件结果集
warning_results = []
warning_results.append("warning:mayun")
warning_results.append("warning:likai")

i = 0
normal_known_faces = []

while(1 == 1):
    try:
        # 加载已知图片
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "normal/image"+str(i)+".jpg"))
        known_image = face_recognition.load_image_file(path)
        # 对图片进行编码，获取128维特征向量
        image_encoding = face_recognition.face_encodings(known_image)[0]
        # # 存为数组以便之后识别
        normal_known_faces.append(image_encoding)
        i = i + 1
    except(FileNotFoundError):
        # print("正常文件扫描结束")
        break

j = 0
warning_known_faces = []
while(1 == 1):
    try:
        # 加载已知图片
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "warning/image"+str(j)+".jpg"))
        warning_known_image = face_recognition.load_image_file(path)
        # 对图片进行编码，获取128维特征向量
        warning_image_encoding = face_recognition.face_encodings(warning_known_image)[0]
        # # 存为数组以便之后识别
        warning_known_faces.append(warning_image_encoding)
        j  = j + 1
    except(FileNotFoundError):
        # print("警告文件扫描结束")
        break

# 加载待识别图片
unknown_image = face_recognition.load_image_file(location_1)

# 初始化一些变量
face_locations = []
face_encodings = []
frame_number = 0

# 获取人脸区域位置
face_locations = face_recognition.face_locations(unknown_image)
# 对图片进行编码，获取128维特征向量
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

if len(face_encodings) == 0:
    print ("Unknown") 

for face_encoding in face_encodings:
    result = None
    # 识别图片中人脸是否匹配已知图片
    warning_match = face_recognition.compare_faces(warning_known_faces, face_encoding, tolerance=0.5)
    k = 0
    result = 'Unknown'
    for match in warning_match:
        if match:
            result = warning_results[k]
        k = k + 1

    if result == 'Unknown':
        normal_match = face_recognition.compare_faces(normal_known_faces, face_encoding, tolerance=0.5)
        k = 0
        for match in normal_match:
            if match:
                result = normal_results[k]
            k = k + 1

    print(result)