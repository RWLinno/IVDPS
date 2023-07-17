import cv2
import torch
import torch.nn as nn
import torchvision
import numpy as np
import time
from PIL import Image
from myResNet import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

model.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
model.add_module("resnet_block2", resnet_block(64, 128, 2))
model.add_module("resnet_block3", resnet_block(128, 256, 2))
model.add_module("resnet_block4", resnet_block(256, 512, 2))
model.add_module("global_avg_pool", GlobalAvgPool2d())
model.add_module("fc", nn.Sequential(Reshape(), nn.Linear(512, 7)))

model.load_state_dict(torch.load("models/model_ResNet_200epoches.pth", map_location=device))
#model.load_state_dict(torch.load("model_vgg_epoches100.pth", map_location=device))
model.eval()

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.Resize((48,48)),
    torchvision.transforms.ToTensor()
])#定义图像变换以符合网络输入

emotion = ["angry","disgust","fear","happy","sad","surprised","neutral"]#表情标签

class Emotion_Recognition_v2():
    def __init__(self):
        cap = cv2.VideoCapture(0)# 摄像头，0是笔记本自带摄像头
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #opencv自带的一个面部识别分类器

        while(cap.isOpened()):
            ret, frame = cap.read()
            frame = frame[:,::-1,:]#水平翻转，符合自拍习惯
            frame= frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
            face = face_cascade.detectMultiScale(gray,1.1,3)  
            img = frame
            if(len(face)>=1):
                (x,y,w,h)= face[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                img = frame[:][y:y+h,x:x+w]  
            # 如果分类器能捕捉到人脸，就对其进行剪裁送入网络，否则就将整张图片送入 
            img = Image.fromarray(img)
            raw = img
            img = transforms(img)
            img = img.reshape(1,1,48,48)
            pre = model(img).max(1)[1].item()
            frame = cv2.putText(frame, emotion[pre], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (55,255,155), 2)
            #显示窗口第一个参数是窗口名，第二个参数是内容
            cv2.imshow('Face real-time expression recognition - press B to quit and S to screenshot', frame)
            if cv2.waitKey(100) == ord('b'):
                break
            elif cv2.waitKey(100) == ord('s'):
                t = time.localtime() #获取时间然后将年月日时分秒连起来
                path = str(t.tm_year)+str(t.tm_mon)+str(t.tm_mday)+str(t.tm_hour)+str(t.tm_min)+str(t.tm_sec) + ".jpg"
                raw.save(path)

        cap.release()
        cv2.destroyAllWindows()