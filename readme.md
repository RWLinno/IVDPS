# IVDPS

Integrated Visual-data Processing System 可视数据综合处理系统

目前是第一个版本，用来做学校的人脸实时表情识别系统，之后会更新比方说图像分类、图像分割这些内容。

Copyright By RWLinno



### Sources文件夹

**main是程序的运行入口**，使用IDE可以运行，运行错误请看项目文档的环境要求。

pic 文件夹为用于数据分析和处理的图片，.ipynb_checkpoints文件夹和\__pycache__文件夹存储IDE的临时文件。

models文件夹里面存放的是预训练的代码和存下里的模型。**fer2013.csv**为模型预训练用到的数据集，文件太大了请自行去Kaggle下载并放入models目录中以供训练使用：https://www.kaggle.com/datasets/deadskull7/fer2013

**haarcascade_frontalface_default.xml**为opencv自带的人脸分类器

ipynb后缀文件是各种模型的**预训练代码**，可以用jupyter notebook打开

pth后缀文件为保存下来的模型，命名格式是"model+模型名+训练批次.pth"

video.py和camera.py里面用的是model_ResNet_200epoches.pth，大家可以根据需要自行修改模型。



### 其他

**说明文档**：有一份pdf有一份doc，防止格式错误。一共89页，附录放了代码。

**PPT:** 有一份pdf一份pptx，防止字体丢失。