import cv2
import torch
import torch.nn as nn
from torchvision.transforms import transforms
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

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 设置图像预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])


class Emotion_Recognition_video():
    def __init__(self,path):
        video_path = path
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #opencv自带的一个面部识别分类器
        video = cv2.VideoCapture(video_path)
        # 开始逐帧预测
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            # 转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = face_cascade.detectMultiScale(gray,1.1,3)  
            if(len(face)>=1):
                (x,y,w,h)= face[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # 图像预处理
            input_data = transform(gray).unsqueeze(0)
            # 运行模型进行预测
            output = model(input_data)
            _, predicted = torch.max(output.data, 1)
            emotion = labels[predicted.item()]
            # 在视频帧上绘制预测结果
            cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Video Real-time Emotion Prediction - press B to quit ', frame)
            # 按下'q'键退出循环
            if cv2.waitKey(1) & 0xFF == ord('b'):
                break

        # 关闭视频和窗口
        video.release()
        cv2.destroyAllWindows()