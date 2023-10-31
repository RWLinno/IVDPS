
#RWLinno's Integrated Visual-data Processing System
import math
import os.path
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.Qt import *
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from scipy import ndimage
import demo_emotion_recognition as ER
import video
import camera
import re
import json
import jsonpath
import tips
import Transform

# 记录图片路径的前驱后继
pre = dict()
nxt = dict()

class Window(QWidget):
# 初始化页面
	def __init__(self):
		super().__init__()
		self.lst_oper = "null" # 记录最后一个操作名称
		self.setWindowTitle(tips.title)
		self.resize(1200,900)

		# 定义字体
		font1 = QFont()
		font1.setFamily('宋体')
		font1.setBold(True)
		font1.setPointSize(16)
		font1.setWeight(20)

#################################特殊功能#################################
		btn_video_emotion = QPushButton(self)
		btn_video_emotion.setText('视频表情识别')
		btn_video_emotion.setFont(font1)  #载入字体设置
		btn_video_emotion.setGeometry(600,32,160,32) #（x坐标，y坐标，宽，高）

		btn_real_time_emotion = QPushButton(self)
		btn_real_time_emotion.setText('摄像头表情识别')
		btn_real_time_emotion.setFont(font1)  #载入字体设置
		btn_real_time_emotion.setGeometry(800,32,160,32) #（x坐标，y坐标，宽，高）

		btn_emotion_recognition = QPushButton(self)
		btn_emotion_recognition.setText('多模态情感分析')
		btn_emotion_recognition.setFont(font1)  #载入字体设置
		btn_emotion_recognition.setGeometry(1000,32,160,32) #（x坐标，y坐标，宽，高）

#################################菜单第一栏#################################
	# 打开图片按钮
		btn_open = QPushButton(self)
		btn_open.setText('打开图片')
		btn_open.move(0, 0)
		btn_open.installEventFilter(self)
	# 预览图片按钮
		btn_view = QPushButton(self)
		btn_view.setText('预览图片')
		btn_view.move(0, 24)
	# 保存图片按钮
		btn_save = QPushButton(self)
		btn_save.setText('保存图片')
		btn_save.move(0, 48)
	# 退出程序按钮
		btn_exit = QPushButton(self)
		btn_exit.setText('退出系统')
		btn_exit.move(0, 72)

##################################菜单第二栏################################
	# 撤销按钮
		btn_pre = QPushButton(self)
		btn_pre.setText('回退')
		btn_pre.move(72, 0)
	# 前进按钮
		btn_nxt = QPushButton(self)
		btn_nxt.setText('前进')
		btn_nxt.move(72, 24)
	# 重命名图片按钮
		btn_rename = QPushButton(self)
		btn_rename.setText('重命名')
		btn_rename.move(72, 48)
	# 灰度变换按钮
		btn_GreyScale = QPushButton(self)
		btn_GreyScale.setText('灰度变换')
		btn_GreyScale.move(72, 72)

##################################菜单第三栏################################
	# 重置画布大小
		btn_resize = QPushButton(self)
		btn_resize.setText('调整画布')
		btn_resize.move(144, 0)
	# 裁剪图片
		btn_crop = QPushButton(self)
		btn_crop.setText('裁剪图片')
		btn_crop.move(144, 24)
	# 拉伸图片按钮
		btn_picresize = QPushButton(self)
		btn_picresize.setText('拉伸图片')
		btn_picresize.move(144, 48)
	# 平移按钮
		btn_translation = QPushButton(self)
		btn_translation.setText('平移图片')
		btn_translation.move(144, 72)

##################################菜单第四栏################################
	# 错切图片按钮
		btn_shear = QPushButton(self)
		btn_shear.setText('错切图片')
		btn_shear.move(216, 0)
	# 镜像翻转按钮
		btn_flip = QPushButton(self)
		btn_flip.setText('镜像翻转')
		btn_flip.move(216, 24)
	# 旋转变换按钮
		btn_rotate = QPushButton(self)
		btn_rotate.setText('旋转图片')
		btn_rotate.move(216, 48)
	#透视变换按钮
		btn_PerspectiveTransform = QPushButton(self)
		btn_PerspectiveTransform.setText('透视变换')
		btn_PerspectiveTransform.move(216, 72)

##################################菜单第五栏###############################
	#线性变换按钮
		btn_LinearTransformation = QPushButton(self)
		btn_LinearTransformation.setText('四则运算')
		btn_LinearTransformation.move(288, 0)
	#图片叠加按钮
		btn_fusion = QPushButton(self)
		btn_fusion.setText('图片叠加')
		btn_fusion.move(288, 24)
	# 阈值分割按钮
		btn_threshold = QPushButton(self)
		btn_threshold.setText('阈值分割')
		btn_threshold.move(288, 48)
	# 边缘检测按钮
		btn_EdgeDetection = QPushButton(self)
		btn_EdgeDetection.setText('边缘检测')
		btn_EdgeDetection.move(288, 72)

##################################菜单第五栏##############################
	# 中值滤波按钮
		btn_medianblur = QPushButton(self)
		btn_medianblur.setText('中值滤波')
		btn_medianblur.move(360, 0)
	# 均值滤波按钮
		btn_meanblur = QPushButton(self)
		btn_meanblur.setText('均值滤波')
		btn_meanblur.move(360, 24)
	# 双边滤波按钮
		btn_bilateral = QPushButton(self)
		btn_bilateral.setText('双边滤波')
		btn_bilateral.move(360, 48)
	# 高低通滤波按钮
		btn_gauss = QPushButton(self)
		btn_gauss.setText('高斯模糊')
		btn_gauss.move(360, 72)

##################################菜单第六栏################################
	#处理图片的窗口,最大为1200*800
		self.pic = QLabel(self)
		self.pic.setFixedSize(1200,800)
		self.pic.move(0,100)
		self.pic.setStyleSheet("QLabel{"
								"background: grey;"
								"color:rgb(255,255,255,120);"
							 	"}")

	#提示文字
		self.tip = QLabel(self)
		self.tip.setStyleSheet("QLabel{"
								"font-size:12px;"
							    "font-family:宋体;}"
								"font-weight:bold;"
								"}")
		self.tip.setText("操作前请打开需要处理的图片,当前画布为1280*900:")
		self.tip.move(600,0)
		self.tip.setFixedSize(800,24)

		self.ope = QLabel(self)
		self.ope.setStyleSheet("QLabel{"
								"font-size:12px;"
							    "font-family:宋体;}"
								"font-weight:bold;"
								"}")
		self.ope.setText("")
		self.ope.move(600,24)
		self.ope.setFixedSize(800,24)
##################################参数栏################################
	#每个参数间隔 Width:120
		self.lab1 = QLabel(self)
		self.lab1.setText("参数1：")
		self.lab1.move(600,75)
		self.par1 = QtWidgets.QLineEdit(self)
		self.par1.setText("")
		self.par1.setGeometry(QtCore.QRect(630,70,70,20))
		self.par1.setObjectName("Parameter1")

		self.lab2 = QLabel(self)
		self.lab2.setText("参数2：")
		self.lab2.move(720,75)
		self.par2 = QtWidgets.QLineEdit(self)
		self.par2.setText("")
		self.par2.setGeometry(QtCore.QRect(750,70,70,20))
		self.par2.setObjectName("Parameter2")

		self.lab3 = QLabel(self)
		self.lab3.setText("参数3：")
		self.lab3.move(840,75)
		self.par3 = QtWidgets.QLineEdit(self)
		self.par3.setGeometry(QtCore.QRect(870,70,70,20))
		self.par3.setText("")
		self.par3.setObjectName("Parameter3")

		self.lab4 = QLabel(self)
		self.lab4.setText("参数4：")
		self.lab4.move(960,75)
		self.par4 = QtWidgets.QLineEdit(self)
		self.par4.setGeometry(QtCore.QRect(990,70,70,20))
		self.par4.setText("")
		self.par4.setObjectName("Parameter4")

		self.lab5 = QLabel(self)
		self.lab5.setText("参数5：")
		self.lab5.move(1080,75)
		self.par5 = QtWidgets.QLineEdit(self)
		self.par5.setGeometry(QtCore.QRect(1110,70,70,20))
		self.par5.setText("filename")
		self.par5.setObjectName("Parameter5")

##################################保存设置栏################################
		self.savetype1 = QtWidgets.QRadioButton(self)
		self.savetype1.setGeometry(QtCore.QRect(450,0,144,24))
		self.savetype1.setText("以时间编号为后缀保存")
		self.savetype1.setChecked(1)
		self.savetype2 = QtWidgets.QRadioButton(self)
		self.savetype2.setGeometry(QtCore.QRect(450,24,144,24))
		self.savetype2.setText("以操作序列为后缀保存")
		self.savetype3 = QtWidgets.QRadioButton(self)
		self.savetype3.setGeometry(QtCore.QRect(450,48,144,24))
		self.savetype3.setText("自定义参数为后缀保存")
		self.savetype4 = QtWidgets.QRadioButton(self)
		self.savetype4.setGeometry(QtCore.QRect(450, 72, 144, 24))
		self.savetype4.setText("覆盖原图")

####################################################################################################
# 基本功能实现部分
####################################################################################################
	# 检查4个参数
	# 0:不检查, 1：需要整数, 2:需要0~1的浮点数, 3:需要一个整数坐标, 4：-180~180的角度, 5：任意浮点数
		def check(type):
			flag = 1
			for i in range(4):
				if type[i] == 0: # 不需要检查
					continue
				str = "$"  # 用来判是否所有参数有无
				if i == 0 and len(self.par1.text()) > 0:
					str = self.par1.text()
				elif i == 1 and len(self.par2.text()) > 0:
					str = self.par2.text()
				elif i == 2 and len(self.par3.text()) > 0:
					str = self.par3.text()
				elif i == 3 and len(self.par4.text()) > 0:
					str = self.par4.text()
				print(str)
				if str == "$":
					return False
				if type[i] == 1:
					if not str.isdigit():
						flag = 0
				elif type[i] == 2:
					s = str.split('.')
					if (len(s) > 2):
						flag = 0
					else:
						for si in s:
							if not si.isdigit():
								flag = 0
						if abs(float(str)) > 1:
							flag = 0
				elif type[i] == 3:
					s = str.split(',')
					if len(s) > 2:
						flag = 0
					else:
						for si in s:
							if not si.isdigit():
								flag = 0
				elif type[i] == 4:
					s = str.split('.')
					if len(s) > 2:
						flag = 0
					else:
						for si in s:
							if not si.isdigit():
								flag = 0
						if abs(float(str)) > 360:
							flag = 0
				elif type[i]== 5:
					if not str.isdigit() :
						flag = 0
					elif int(str) != 0 and int(str) != 1 :
						flag = 1

				if flag == 0:
					return False
			return True

	# 获取时间编号
		def rwl_time():
			t = time.localtime() #获取时间然后将年月日时分秒连起来
			return str(t.tm_year)+str(t.tm_mon)+str(t.tm_mday)+str(t.tm_hour)+str(t.tm_min)+str(t.tm_sec)

	# 撤销操作
		def rwl_pre():
			print("pre!!!")
			if self.imgpath in pre.keys() :
				self.imgpath = pre[self.imgpath]  # 通过字典实现的双线链表实现撤回
				s="图片路径："+str(self.imgpath)
				self.tip.setText(s)
				rwl_refresh() # 刷新窗口
			else :
				self.tip.setText("前一个操作不存在!!!")
		btn_pre.clicked.connect(rwl_pre)

	# 前进操作
		def rwl_nxt():
			print("nxt!!!")
			if self.imgpath in nxt.keys() :
				self.imgpath = nxt[self.imgpath]
				s = "图片路径："+str(self.imgpath)
				self.tip.setText(s)
				rwl_refresh() # 刷新窗口
			else :
				self.tip.setText("后一个操作不存在!!!")
		btn_nxt.clicked.connect(rwl_nxt)

	# 重命名图片
		def rwl_rename():
			filename = os.path.basename(self.imgpath) # 获取文件名
			path = self.imgpath.strip(filename) # 获取目录路径
			dic = filename.split(".")
			tmp_str = str(self.par5.text())
			new_path = str(path) + str(tmp_str) + "." + str(dic[1])
			nxt[self.imgpath] = new_path # 建立双向链表
			pre[new_path] = self.imgpath  # 保存上一张图片的地址
			self.imgpath = new_path  # 更改到最新的图片的地址
			cv2.imwrite(new_path,self.now_pic) # 保存为新命名的图片
			QMessageBox.information(window,"操作成功","成功另存为:"+new_path)
			s = "图片路径：" + str(self.imgpath)
			self.tip.setText(s)
			rwl_refresh()
		btn_rename.clicked.connect(rwl_rename)

	#保存图像
		def rwl_save():
			path=self.imgpath.split('.')
			if self.savetype1.isChecked() :  # 以时间编号作为后缀
				tmp_str = "_" + str(rwl_time())
			elif self.savetype2.isChecked(): # 以操作序列作为后缀
				tmp_str = "_" + self.lst_oper
			elif self.savetype3.isChecked(): # 以参数5作为后缀
				tmp_str = str(self.par5.text())
			elif self.savetype4.isChecked(): # 覆盖原图片
				tmp_str = ""
			s=str(path[0]) + tmp_str + "." + str(path[1])
			if(self.imgpath != s) :
				nxt[self.imgpath] = s
				pre[s] = self.imgpath  # 保存上一张图片的地址

			self.imgpath = s   # 更改到最新的图片的地址
			cv2.imwrite(s,self.now_pic)
			QMessageBox.information(window,"操作成功","成功保存到:"+s)
			rwl_refresh()
			return
		btn_save.clicked.connect(rwl_save)

	#预览图片
		def rwl_view():
			plt.imshow(cv2.cvtColor(self.now_pic, cv2.COLOR_BGR2RGB))
			plt.show()
		btn_view.clicked.connect(rwl_view)

		def gets_json(obj,name):
			value = jsonpath.jsonpath(obj, "$.."+str(name))[0]
			return value

	# 图片情感分析
		def rwl_emotion_recognition():
			obj = ER.Emotion_Recognition_v1(self.imgpath)
			qront = obj.recognize_emotion()
			emotion = gets_json(qront,"emotion")
			#face_num = gets_json(qront,"face_num")
			print("emotion=",emotion)
		#	print("face_num=",face_num)
			print(type(emotion))
			anger = emotion['anger']
			disgust = emotion['disgust']
			fear = emotion['fear']
			happiness = emotion['happiness']
			neutral = emotion['neutral']
			sadness = emotion['sadness']	
			surprise = emotion['surprise']
			
			values=[anger,disgust,fear,happiness,neutral,sadness,surprise]
			label=['anger','dismay','fear','joy','neutrality','sadness',' excitement']
			colors = ["#C7EDCC","#FAF9DE","#FFF2E2","#FDE6E0","#E3EDCD","#DCE2F1","#E9EBFE"]
			explode = [0.2, 0, 0, 0, 0, 0, 0]
			# 绘制饼状图
			plt.title('Emotion Distribution')
			ax = plt.subplot(121)
			wedges,texts,autotexts = plt.pie(values,
                                 autopct="%3.1f%%",
                                 textprops=dict(color = "black"),
                                 colors = colors,
								 explode=explode,
								 pctdistance=2)
			plt.legend(wedges,label,fontsize = 12,loc = "center left",bbox_to_anchor=(0.95,0,0.3,1))
			# 绘制柱状图
			plt.subplot(122)
			plt.barh(label, values)
			# 调整子图布局
			#plt.ylim(0, max(values) * 1.2)  # 根据最大值调整y轴范围
			# 显示图形
			plt.gca().invert_xaxis()  # 反转y轴，使标签放在右边
			plt.xlim(0, max(values) * 1.2)  # 根据最大值调整x轴范围
			plt.yticks(range(len(label)), label)
			plt.gca().set_yticklabels(label, ha='right')
			plt.savefig('./Emotional visualization')#保存图片
			# 调整子图布局
			plt.subplots_adjust(wspace=2.5)  # 调整子图之间的水平间距
			plt.show()
			return
		btn_emotion_recognition.clicked.connect(rwl_emotion_recognition)

		def rwl_real_time_emotion():
			camera.Emotion_Recognition_v2()
			return
		btn_real_time_emotion.clicked.connect(rwl_real_time_emotion)
		
		def rwl_video_emotion():
			directory = QFileDialog.getOpenFileName(self,
													"getOpenFileName", "./",
													"All Files (*);;Text Files (*.txt)")
			video_path = directory[0] 			# 用控件获取视频路径
			video.Emotion_Recognition_video(video_path)
			return
		btn_video_emotion.clicked.connect(rwl_video_emotion)

	#刷新图像
		def rwl_refresh():
			print(self.imgpath)
			# 记录上一个操作和参数
			s = self.lst_oper+" "+self.par1.text()+" "+self.par2.text()+" "+self.par3.text()+" "+self.par4.text()
			self.ope.setText(s)
			# 重新读取图片
			self.pmap = QPixmap(self.imgpath)
			self.now_pic = cv2.imread(self.imgpath,0)
			self.pic.setPixmap(self.pmap)
			return

	#退出系统
		def rwl_exit():
			select = QMessageBox.information(self, '退出系统', '是否保存图片', QMessageBox.Yes | QMessageBox.No)
			if select == QMessageBox.Yes:
				rwl_save()
			exit(0)
		btn_exit.clicked.connect(rwl_exit)

	#打开图片
		def rwl_open_pic():
			# 用控件获取路径
			directory = QFileDialog.getOpenFileName(self,
													"getOpenFileName", "./",
													"All Files (*);;Text Files (*.txt)")
			self.imgpath = directory[0]
			self.pmap = QPixmap(self.imgpath)
			self.now_pic = cv2.imread(self.imgpath)
			#self.now_pic = cv2.cvtColor(self.now_pic,cv2.COLOR_BGR2RGB)
			self.pic.setPixmap(self.pmap)
			# 更新提示信息
			s = "图片大小为:" + str(self.pmap.width()) + "x" + str(self.pmap.height()) + ",开始数字图像处理吧！"
			self.tip.setText(s)
		btn_open.clicked.connect(rwl_open_pic)

	# 灰度变换
		def rwl_GreyScale():
			img = self.now_pic
			rows, cols = img.shape[:2]
			x = 0
			y = 0
			z = 0
			if check([1,1,1,0])== False :
				QMessageBox.information(self, tips.GreyScale_error, QMessageBox.Ok)
				return
			if len(self.par1.text()) != 0:
				x = int(self.par1.text())
			if len(self.par2.text()) != 0:
				y = int(self.par2.text())
			if len(self.par3.text()) != 0:
				z = int(self.par3.text())
			mx = img.max()
			window.setCursor(Qt.WaitCursor)
			res1 = img.copy()  # 翻转变换结果
			res2 = img.copy()  # 对数变换结果
			res3 = img.copy()  # 伽马变换结果
			res4 = Equalization(img)  # 直方图均衡化
			for i in range(rows):
				for j in range(cols):
					r = img[i,j]
					res1[i,j] = mx - r
					res2[i,j] = ((y*math.log(1 + r) - y * math.log(1 + 0))/\
                                (y*math.log(1 + mx) - y*math.log(1 + 0))) * mx
					res3[i,j] = math.pow(r/mx,z)*mx
			if x==1 :
				self.now_pic = res1
			elif x==2 :
				self.now_pic = res2
			elif x==3 :
				self.now_pic = res3
			else :
				# res4 = Equalization(img) # 直方图均衡化
				self.now_pic = res4
			self.lst_oper = "grey"
			rwl_save()
			window.setCursor(Qt.ArrowCursor)
			print("灰度变换成功！！！")
		btn_GreyScale.clicked.connect(rwl_GreyScale)

###############################################################################################
	# 调整画布
		def rwl_resize():
			if check([1,1,0,0])== False :
				QMessageBox.information(self, tips.Resize_error, QMessageBox.Ok)
				return
			x = 1200
			y = 800
			if len(self.par1.text()) != 0 :
				x = int(self.par1.text())
				if x == 0:
					x=1200
			if len(self.par2.text()) != 0 :
				y = int(self.par2.text())
				if y == 0:
					y = 800
			self.resize(max(1200,x), max(900,y+100)) # 画布不能太小
			self.pic.setFixedSize(x,y)
		btn_resize.clicked.connect(rwl_resize)

	# 裁剪图像
		def rwl_crop():
			img = self.now_pic
			rows, cols = img.shape[:2]
			x = 0
			y = 0
			z = 0
			w = 0
			if check([1,1,1,1])== False :
				QMessageBox.information(self,tips.Crop_error, QMessageBox.Ok)
				return
			if len(self.par1.text()) != 0 :
				x = int(self.par1.text())
			if len(self.par2.text()) != 0 :
				y = int(self.par2.text())
				if y == 0:
					y = rows
			if len(self.par3.text()) != 0 :
				z = int(self.par3.text())
			if len(self.par4.text()) != 0 :
				w = int(self.par4.text())
				if w == 0:
					w = cols
			res = self.now_pic[z:w,x:y] # 直接截取灰度矩阵的某一个区域
			self.now_pic =res
			self.lst_oper = "crop"
			rwl_save()
			#rwl_refresh()
			print("裁剪成功!!!")
		btn_crop.clicked.connect(rwl_crop)

	# 拉伸图片
		def rwl_picresize():
			if check([1,1,0,0])== False :
				QMessageBox.information(self, tips.Picresize_error, QMessageBox.Ok)
				return
			rows , cols = self.now_pic.shape
			if len(self.par1.text()) != 0 :
				h = int(self.par1.text())
				if h == 0:
					h = rows
			if len(self.par2.text()) != 0 :
				w = int(self.par2.text())
				if w == 0:
					w = cols
			select = QMessageBox.information(self, '退出系统', '请选择填充方式,Yes:最邻近插值/No:双线性插值', QMessageBox.Yes | QMessageBox.No)
			if select == QMessageBox.Yes :
				# 最邻近插值
				self.now_pic = cv2.resize(self.now_pic, (h, w), interpolation=cv2.INTER_NEAREST)
			else :
				# 双线性插值
				self.now_pic = cv2.resize(self.now_pic, (h, w), interpolation=cv2.INTER_LINEAR_EXACT)
			self.lst_oper = "resize"
			rwl_save()
			#rwl_refresh()
			print("插值成功!!!")
		btn_picresize.clicked.connect(rwl_picresize)

	# 平移图片
		def rwl_translation():
			if check([1,1,0,0])== False :
				QMessageBox.information(self, '请确认参数', '无效操作！！！请确认你的参数是否正确。\n[参数1]：向x轴平移的像素值bx\n[参数2]：向y轴平移的像素值by\n将图片平移[bx,by]，如果参数无效，默认为0\n', QMessageBox.Ok)
				return
			rows, cols = self.now_pic.shape[:2]
			# 定义平移矩阵，需要是numpy的float32类型
			bx = 0
			by = 0
			if len(self.par1.text()) != 0 :
				bx = int(self.par1.text())
			if len(self.par2.text()) != 0 :
				by = int(self.par2.text())
			# 变换矩阵
			window.setCursor(Qt.WaitCursor)
			M = np.array([
				[1, 0, bx],
				[0, 1, by]
			],dtype=np.float32)

			res = rwl_WarpAffine(self.now_pic, M, (cols, rows))  # 用仿射变换实现平移
			self.now_pic = res
			self.lst_oper = "translation"
			rwl_save()
			#rwl_refresh()
			window.setCursor(Qt.ArrowCursor)
			print("平移成功!!!")
		btn_translation.clicked.connect(rwl_translation)

###############################################################################################
	# 错切图片
		def rwl_shear():
			if check([2,2,0,0])== False :
				QMessageBox.information(self, tips.Shear_error, QMessageBox.Ok)
				return
			rows, cols = self.now_pic.shape[:2]
			if len(self.par1.text()) != 0 :
				x = float(self.par1.text())
			if len(self.par2.text()) != 0 :
				y = float(self.par2.text())
			# 变换矩阵
			M = np.array([
				[1,x,0],
				[y,1,0]
			],dtype=np.float32)
			window.setCursor(Qt.WaitCursor)
			res = rwl_WarpAffine(self.now_pic,M,(cols,rows))
			self.now_pic = res
			self.lst_oper = "shear"
			rwl_save()
			window.setCursor(Qt.ArrowCursor)
			print("错切成功!!!")
		btn_shear.clicked.connect(rwl_shear)
	# 翻转图片
		def rwl_flip():
			if check([5,5,0,0])== False :
				QMessageBox.information(self, tips.Flip_error, QMessageBox.Ok)
				return
			rows, cols = self.now_pic.shape[:2]
			x = 0
			y = 0
			window.setCursor(Qt.WaitCursor)
			img = self.now_pic
			if len(self.par1.text()) != 0 :
				x = int(self.par1.text())
			if len(self.par2.text()) != 0 :
				y = int(self.par2.text())
			# 水平翻转
			res = self.now_pic
			if x == 1 :
				M = np.array([
					[-1,0,cols],
					[0,1,0]
				],dtype=np.float32)
				res = rwl_WarpAffine(res, M, (cols, rows))
			#垂直翻转
			if y == 1 :
				M = np.array([
					[1, 0, 0],
					[0, -1, rows]
				], dtype=np.float32)
				res = rwl_WarpAffine(res, M, (cols, rows))
			self.now_pic = res
			rwl_save()
			self.lst_oper = "flip"
			window.setCursor(Qt.ArrowCursor)
			print(x,y,"翻转成功!!!")
		btn_flip.clicked.connect(rwl_flip)

	# 旋转图片
		def rwl_rotate():
			if check([4,1,1,0])== False :
				QMessageBox.information(self, tips.Rotate_error, QMessageBox.Ok)
				return
			img = self.now_pic
			rows, cols = img.shape[:2]
			rt = 0.0
			x = 0
			y = 0
			window.setCursor(Qt.WaitCursor)
			if len(self.par1.text()) != 0 :
				rt = float(self.par1.text())
			if len(self.par2.text()) != 0 :
				x = int(self.par2.text())
			if len(self.par3.text()) != 0 :
				y = int(self.par3.text())
			beta = rt*np.pi/180
			# 变换矩阵
			M = np.array([
				[np.cos(beta),np.sin(beta),x],
				[-np.sin(beta),np.cos(beta),y]
			],dtype=np.float32)
			M2 = cv2.getRotationMatrix2D((rows // 2, cols // 2), rt, 1)  #这是绕中心旋转
			res = rwl_WarpAffine(img,M,(cols,rows))
			self.now_pic = res
			self.lst_oper = "rotate"
			rwl_save()
			# rwl_refresh()
			window.setCursor(Qt.ArrowCursor)
			print("旋转成功!!!")
		btn_rotate.clicked.connect(rwl_rotate)

	# 透视变换
		def rwl_PerspectiveTransform():
			if check([3,3,3,3])== False :
				QMessageBox.information(self, '透视变换', tips.PerspectiveTransform_error, QMessageBox.Ok)
				return
			img = self.now_pic
			h, w = img.shape[:2]
			x1, y1, x2, y2, x3, y3, x4, y4 = map(int, self.par1.text().split(',') + self.par2.text().split(',') +
                                    self.par3.text().split(',') + self.par4.text().split(','))

			window.setCursor(Qt.WaitCursor)
			src = np.array([
				[x1, y1],
				[x2, y2],
				[x3, y3],
				[x4, y4]
			], dtype=np.float32)
			dst = np.array([
				[0, 0],
				[0, w],
				[h, w],
				[h, 0]
			], dtype=np.float32)
			M = Transform.getPerspectiveTransform(src, dst)
			#res = Transform.warpPerspective(img, M, (w, h)) #这里不会实现
			#M = cv2.getPerspectiveTransform(src, dst)
			res = cv2.warpPerspective(img, M, (w, h))
			self.now_pic = res
			self.lst_oper = "PerspectiveTransform"
			rwl_save()
			# rwl_refresh()
			window.setCursor(Qt.ArrowCursor)
			print("透视变换成功!!!")
		btn_PerspectiveTransform.clicked.connect(rwl_PerspectiveTransform)

###############################################################################################
	# 四则运算部分
		def rwl_LinearTransformation():
			if check([1,1,2,2])== False :
				QMessageBox.information(self, '线性变换', tips.LinearTransformation_error , QMessageBox.Ok)
				return
			window.setCursor(Qt.WaitCursor)
			if len(self.par1.text()) != 0:
				if int(self.par1.text()) != 0:
					self.now_pic = self.now_pic + float(self.par1.text())
			img_add = self.now_pic
			if len(self.par2.text()) != 0:
				if int(self.par2.text()) != 0:
					self.now_pic = self.now_pic - float(self.par2.text())
			img_sub = self.now_pic
			if len(self.par3.text())!= 0:
				if float(self.par3.text()) != 0:
					self.now_pic = self.now_pic * float(self.par3.text())
			img_mul = self.now_pic
			if len(self.par4.text()) != 0:
				if float(self.par4.text())!=0 :
					self.now_pic = self.now_pic / float(self.par4.text())
			img_div = self.now_pic
			self.lst_oper = "LinearTransformation"
			rwl_save()
			window.setCursor(Qt.ArrowCursor)
			# rwl_refresh()
			print("四则成功!!!")
		btn_LinearTransformation.clicked.connect(rwl_LinearTransformation)

	# 图片叠加部分
		def rwl_fusion():
			if check([2,2,0,0])== False :
				QMessageBox.information(self, tips.Fusion_error, QMessageBox.Ok)
				return
			# 打开一张新的图片
			directory = QFileDialog.getOpenFileName(self,
													"getOpenFileName", "./",
													"All Files (*);;Text Files (*.txt)")
			img = self.now_pic
			path = directory[0]
			obj = cv2.imread(path,0)
			# 先将图片转成灰度图
			#if obj.ndim != 2 :
			#	obj = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
			h, w = img.shape[:2]
			x = 0
			y = 0
			window.setCursor(Qt.WaitCursor)
			if len(self.par1.text()) != 0:
				x = float(self.par1.text())
				if x == 0:
					x = 0.5
				img = img * x

			if len(self.par2.text()) != 0:
				y = float(self.par2.text())
				if y == 0:
					y = 0.5
				obj = obj * y
			# 调成一样的大小
			obj.resize(h, w)
			res = (img + obj)
			self.now_pic = res
			self.lst_oper = "fusion"
			rwl_save()
			window.setCursor(Qt.ArrowCursor)
			print("图片融合成功!!!")
		btn_fusion.clicked.connect(rwl_fusion)

	# 阈值分割部分
		def rwl_threshold():
			if check([2,5,0,0])== False :
				QMessageBox.information(self, tips.Threshold_error, QMessageBox.Ok)
				return
			x = 0
			y = 0
			window.setCursor(Qt.WaitCursor)
			if len(self.par1.text()) != 0 :
				y = float(self.par1.text())
			if len(self.par2.text()) != 0 :
				x = int(self.par2.text())
			img = self.now_pic
			res1 = Threshold(img, get_T1(img), y)
			res2 = Threshold(img, get_T2(img), y)
			if x == 1 :
				self.now_pic = res1
			else :
				self.now_pic = res2
			self.lst_oper = "threshold"
			rwl_save()
			# rwl_refresh()
			window.setCursor(Qt.ArrowCursor)
			print("阈值分割成功!!!")
		btn_threshold.clicked.connect(rwl_threshold)

	# 边缘检测
		def rwl_EdgeDetection():
			if check([1,0,0,0])== False :
				QMessageBox.information(self, tips.EdgeDetection_error, QMessageBox.Ok)
				return
			x = 3
			window.setCursor(Qt.WaitCursor) # 设置等待光标
			if len(self.par1.text()) != 0:
				x = int(self.par1.text())
			img = self.now_pic
			h, w = img.shape[:2]
			res1 = Sobel(img) # 使用Sobel算子实现
			res2 = Robert(img) # 使用Robert算子实现
			res3 = Laplace(img) # 使用Laplace算子实现
			if x == 1:
				self.now_pic = res1
			elif x == 2:
				self.now_pic = res2
			else:
				self.now_pic = res3
			self.lst_oper = "EdgeDetection"
			rwl_save()
			#rwl_refresh()
			window.setCursor(Qt.ArrowCursor) #光标换回原来的
			print("边缘检测成功！！！")
		btn_EdgeDetection.clicked.connect(rwl_EdgeDetection)
###############################################################################################
	# 中值滤波部分
		def rwl_medianblur():
			if check([1,0,0,0])== False :
				QMessageBox.information(self, tips.Medianblur_error, QMessageBox.Ok)
				return
			img = self.now_pic
			h, w = img.shape[:2]
			x = 0
			window.setCursor(Qt.WaitCursor)
			if len(self.par1.text()) != 0 :
				x = int(self.par1.text())
				if x < 3:
					x = 5
			res= MedianBlur(img,x)
			self.now_pic = res[0:w - x, 0:h - x]
			self.lst_oper = "medianblur"
			rwl_save()
			# rwl_refresh()
			window.setCursor(self.Qt.ArrowCursor)
		btn_medianblur.clicked.connect(rwl_medianblur)

	# 均值滤波部分
		def rwl_meanblur():
			if check([1, 0, 0, 0]) == False:
				self.information(self, tips.MeanBlur_error, QMessageBox.Ok)
				return
			img = self.now_pic
			h, w = img.shape[:2]
			x = 3
			window.setCursor(Qt.WaitCursor)
			if len(self.par1.text()) != 0:
				x = int(self.par1.text())
				if x < 3:
					x = 3
			res = MeanBlur(img, x)
			self.now_pic = res[0:w - x, 0:h - x]
			self.lst_oper = "meanblur"
			rwl_save()
			# rwl_refresh()
			window.setCursor(Qt.ArrowCursor)
		btn_meanblur.clicked.connect(rwl_meanblur)

	# 双边滤波部分
		def rwl_bilateral():
			if check([1, 1, 1, 0]) == False:
				QMessageBox.information(self, tips.Bilateral_error, QMessageBox.Ok)
				return
			img = self.now_pic
			h, w =img.shape[:2]
			x = 20
			y = 20
			z = 5
			window.setCursor(Qt.WaitCursor)
			if len(self.par1.text()) != 0 :
				x = int(self.par1.text())
			if len(self.par2.text()) != 0 :
				y = int(self.par2.text())
			if len(self.par3.text()) != 0 :
				z = int(self.par3.text())
			res = Bilateral(img,x,y,z)
			self.now_pic = res[0:w-z, 0:h-z]
			self.lst_oper = "bilateral"
			rwl_save()
			# rwl_refresh()
			window.setCursor(Qt.ArrowCursor)
		btn_bilateral.clicked.connect(rwl_bilateral)

	# 高斯模糊
		def rwl_gauss():
			if check([5, 1, 0, 0]) == False:
				QMessageBox.information(self, tips.Gauss_error, QMessageBox.Ok)
				return
			img = self.now_pic
			h, w = img.shape[:2]
			x = 20
			y = 3
			window.setCursor(Qt.WaitCursor)
			if len(self.par1.text()) != 0:
				x = float(self.par1.text())
				if x<5:
					x = 5
			if len(self.par2.text()) != 0:
				y = int(self.par2.text())
				if y<3 :
					y = 3
			res = gauss(img,y,x)
			self.now_pic = res[0:w - (2*y+1), 0:h - (2*y+1)]
			self.lst_oper = "gauss"
			rwl_save()
			# rwl_refresh()
			window.setCursor(Qt.ArrowCursor)
		btn_gauss.clicked.connect(rwl_gauss)

####################################################################################################
#手写实现变换算子
####################################################################################################

	# 高斯滤波器
		def gauss(img,k,sigma):
			size = 2 * k + 1
			h,w = img.shape[:2]
			kernel = np.zeros((size,size),np.float32)
			# 首先算出卷积核各单元的值
			for i in range(size):
				for j in range(size) :
					# 高斯密度正态分布
					norm = math.pow(i-k,2)+ math.pow(j-k,2)
					kernel [i,j] = math.exp(-norm/(2*math.pow(sigma,2)))/2*math.pi*pow(sigma,2)
			sum = np.sum(kernel)
			# 最终高度以概率形式呈现
			kernel = kernel / sum
			#print(kernel)
			k_h,k_w = kernel.shape
			for i in range(int(k_h/2),h-int(k_h/2)):
				for j in range(int(k_w/2),w-int(k_w/2)):
					sum = 0
					# 对每个坐标经过高斯滤波算出最终灰度值
					for k in range(0,k_h):
						for l in range(0,k_w):
							sum+=img[i-int(k_h/2)+k,j-int(k_h/2)+l]*kernel[k,l]
					img[i,j] = sum
			#print(img)
			return img

	# 直方图均衡化
		def Equalization(img):
			h, w = img.shape[:2]
			n = h * w
			# 初始化数据
			cnt = [0 for i in range(256)]
			p = [0 for i in range(256)]
			sum = [0 for i in range(256)]
			res = img.copy()
			# 统计灰度出现的次数
			for i in range(w):
				for j in range(h):
					cnt[img[i,j]] += 1
			# 归一化
			for i in range(0,256):
				print(cnt[i],n)
				p[i] = float(cnt[i]/n)
			# 计算累积直方图
			sum[0] = p[0]
			for i in range(1,256):
				sum[i] = sum[i-1] + p[i]
			#计算新的像素值
			for i in range(w):
				for j in range(h):
					res[i,j] = 255 * sum[img[i,j]]
			print("直方图均衡化完成!!!")
			return res
	# 阈值分割
		#大津法获取阈值
		def get_T1(img):
			Sigma = -1
			T = 0
			for t in range(0, 256):
				bg = img[img <= t]
				obj = img[img > t]
				p0 = bg.size / img.size
				p1 = obj.size / img.size
				m0 = 0 if bg.size == 0 else bg.mean()
				m1 = 0 if obj.size == 0 else obj.mean()
				sigma = p0 * p1 * (m0 - m1) ** 2
				if sigma > Sigma:
					Sigma = sigma
					T = t
			return T
		# 迭代法获得阈值
		def get_T2(img):
			T = img.mean() # 以全局均值作为最终停止的标准
			while True:
				t0 = img[img<T].mean()
				t1 = img[img>=T].mean()
				t = (t0+t1) / 2
				if( T == t ): #达到两类像素均值相同
					break
				T=t
			return T
	# 分割的主要过程
		def Threshold(img, T ,X=-1.0):
			h,w = img.shape
			if X == -1:
				res1 = np.uint8( img > T ) * 255 # 二值化，要么0要么255
				return res1

			res2 = img.copy()
			for i in range(h):
				for j in range(w):
					if img[i,j] >= T : #每个像素往两边靠
						res2[i,j] = min(255,img[i,j] + (255-img[i,j]) * X )
					else :
						res2[i,j] = max(0,img[i,j] - img[i,j] * X )
			return res2

	# 中值滤波
		def MedianBlur(img, size):
			h,w = img.shape
			img = img.copy()
			mid = (size - 1)//2
			mdd = [] # 存储卷积核内的每个像素值
			for i in range(h - size):
				for j in range(w - size):
					for m1 in range(size):
						for m2 in range(size):
							mdd.append(int(img[i+m1,j+m2]))
					mdd.sort() # 排序后取中值
					img[i+mid,j+mid] = mdd[(len(mdd)+1)//2]
					del mdd[:] # 清空
			return img

	# 均值滤波
		def MeanBlur(img, size):
			h, w = img.shape
			img = img.copy()
			mid = (size - 1) // 2
			sum = 0
			sz = size * size
			for i in range(h - size):
				for j in range(w - size):
					for m1 in range(size):
						for m2 in range(size):
							sum = sum + int(img[i + m1, j + m2]) # 得到卷积核中所有数总和
					img[i + mid, j + mid] = sum // (sz) #得到均值
					sum = 0
			return img

	# 双边滤波
		# 定义灰度差异卷积核
		def get_C(sigmad, n):
			C = np.zeros((n, n))
			x = np.array([n // 2, n // 2])
			for i in range(n - 1):
				for j in range(n - 1):
					ksi = np.array([i, j])
					C[i, j] = np.exp(-0.5 * (np.linalg.norm(ksi - x) / sigmad) ** 2)
			return C / C.sum()
		# 定义距离差异卷积核
		def get_S(f, sigmar, n):
			f = np.float64(f)  # 这里可以防止负数
			S = np.exp(-0.5 * ((f - f[n // 2, n // 2]) / sigmar) ** 2)
			S /= S.sum()
			return S

		def Bilateral(img, sigmar, sigmad, n):  # 自制双边滤波器
			h, w = img.shape
			res = img.copy()
			C = get_C(sigmad, n)  # 灰度差异滤波器全程不变
			for i in range(h - n):
				for j in range(w - n):
					f = img[i:i + n, j:j + n]  # 原图像区域
					S = get_S(f, sigmar, n)  # 计算空间距离滤波器
					K = C * S
					K /= K.sum()
					res[i, j] = (f * K).sum()
			return res

	# 边缘检测
		# robert算子
		def Robert(img):
			r, c = img.shape
			res=img.copy()
			R = [[-1, -1], [1, 1]]
			for x in range(r):
				for y in range(c):
					if (y + 2 <= c) and (x + 2 <= r):
						imgChild = img[x:x + 2, y:y + 2]
						list_robert = R * imgChild
						res[x, y] = abs(list_robert.sum())  # 求和加绝对值
			return res
		# sobel算子
		def Sobel(img):
			r, c = img.shape
			res = np.zeros((r, c))
			resX = np.zeros(img.shape)
			resY = np.zeros(img.shape)
			SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # X方向
			SY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
			for i in range(r - 2):
				for j in range(c - 2):
					resX[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * SX))
					resY[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * SY))
					res[i + 1, j + 1] = (resX[i + 1, j + 1] * resX[i + 1, j + 1] + resY[
						i + 1, j + 1] * resY[i + 1, j + 1]) ** 0.5
			return np.uint8(res)  # 无方向算子处理的图像
		# Laplace算子
		def Laplace(img):
			r, c = img.shape
			res = np.zeros((r, c))
			L = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
			for i in range(r - 2):
				for j in range(c - 2):
					res[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * L))
			return np.uint8(res)

	# 仿射变换
		def rwl_WarpAffine(img, M, dst_size, constant=(0, 0, 0)):
			M = cv2.invertAffineTransform(M)  # 求仿射变换的逆矩阵
			constant = np.array(constant)
			ih, iw = img.shape[:2]
			dw, dh = dst_size
			dst = np.full((dh, dw, 3), constant, dtype=np.uint8)
			irange = lambda p: p[0] >= 0 and p[0] < iw and p[1] >= 0 and p[1] < ih  # 边界判断
			for y in range(dh):
				for x in range(dw):
					homogeneous = np.array([[x, y, 1]]).T  # 一个列矩阵
					ox, oy = M @ homogeneous  # 把目标的点仿射变换为原始图像的点
					low_ox = int(np.floor(ox))  # 向下取整
					low_oy = int(np.floor(oy))  # 向下取整
					high_ox = low_ox + 1  # 向上取整
					high_oy = low_oy + 1  # 向上取整
					pos = ox - low_ox, oy - low_oy  # 获取相对位置
					p0_area = (1 - pos[0]) * (1 - pos[1])
					p1_area = pos[0] * (1 - pos[1])
					p2_area = (1 - pos[0]) * pos[1]
					p3_area = pos[0] * pos[1]
					p0 = low_ox, low_oy
					p1 = high_ox, low_oy
					p2 = low_ox, high_oy
					p3 = high_ox, high_oy
					p0_value = img[p0[1], p0[0]] if irange(p0) else constant
					p1_value = img[p1[1], p1[0]] if irange(p1) else constant
					p2_value = img[p2[1], p2[0]] if irange(p2) else constant
					p3_value = img[p3[1], p3[0]] if irange(p3) else constant
					dst[y, x] = p0_area * p0_value + p1_area * p1_value + p2_area * p2_value + p3_area * p3_value
			return dst
		# 低通滤波
		def LowPass(h,w,d0,n=2):
			H=np.empty(shape=[h,w],dtype=float)
			mid_x=int(w/2)
			mid_y=int(h/2)
			for y in range(0,h):
				for x in range(0,w):
					d=np.sqrt((x-mid_x)**2+(y-mid_y)**2)
					if d<=d0:
						H[y,x]=1
					else:
						H[y,x]=0
			return H
		# 高通滤波
		def HighPass(h, w, d0, n=2):
			H = np.empty(shape=[h, w], dtype=float)
			mid_x = int(w / 2)
			mid_y = int(h / 2)
			for y in range(0, h):
				for x in range(0, w):
					d = np.sqrt((x - mid_x) ** 2 + (y - mid_y) ** 2)
					if d <= d0:
						H[y, x] = 0
					else:
						H[y, x] = 1
			return H

#程序入口
if __name__ == '__main__':
	app = QApplication(sys.argv)
	window = Window()
	window.show()
	sys.exit(app.exec_())
