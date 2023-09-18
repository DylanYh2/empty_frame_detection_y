import os
import threading
import time
import os
import cv2
import numpy as np
import datetime
import signal
from threading import Thread
import torch
import logging
from predict import predict_carema
from logging.handlers import TimedRotatingFileHandler
from opcua import Client, ua
import argparse
import gc
from train import SELFMODEL
import os
import os.path as osp
import shutil
import torch.nn as nn
from PIL import Image
import sys
import json
sys.path.append("yolov5/") 
from detect import api_run
gc.collect()
device = torch.device('cuda')
infer_dst_path = 'news/'

url = "rtsp://admin:v1234567@192.168.0.110:554/"
first_model_flag = False
sec_model_flag = False
flag1 = False
flag = False
tray_in_pos=False
tray_in_posold=False
tray_flag=False
opc_con_flag=False
# loop_end=False 
count=0
def handle_int(sig, frame):
    print("Ctrl+c press")
    cv2.destroyAllWindows()
    client.disconnect()
    gc.collect()
    os._exit() 

class SubHandler(object):
    def datachange_notification(self, node, val, data):

        node1 = client.get_node('ns=4;i=119')

        dv = ua.DataValue(ua.Variant(val, ua.VariantType.Boolean))
        node1.set_value(dv)
        print("Python: New data change event", node, val)
        
def run(): 
    global tray_in_pos  
    global opc_con_flag     
    global tray_flag    
    global tray_in_posold   #托盘到位信号,默认为False,托盘到位信号为True,托盘不在位信号为False
    # global loop_end
    global count
    try:
        if opc_con_flag:
            node = client.get_node('ns=2;s=Application.OPCUA.ETRS_ETR_L_EM_R.General_TO_ETRS_PLC_LifeBeat') 
            node1 = client.get_node('ns=2;s=Application.OPCUA.ETRS_ETR_L_EM_R.General_FROM_ETRS_ETRS_LifeBeat')
            node2 = client.get_node('ns=2;s=Application.OPCUA.HLC_TPS_L_EM_RFID_R.TO_HLC_CameraStartReq')
            dv = ua.DataValue(ua.Variant(node.get_value(), ua.VariantType.Boolean)) 
            node1.set_value(dv) 
            tray_in_pos = node2.get_value() #获取托盘到位信号,如果托盘到位信号发生变化,则修改托盘到位信号,托盘到位信号为True,托盘不在位信号为False
            if tray_in_pos and not tray_in_posold: #如果托盘到位信号发生变化,则修改托盘到位信号,托盘到位信号为True,托盘不在位信号为False
                tray_flag = True #托盘到位信号为True,托盘不在位信号为False
                # loop_end = False #托盘到位信号为True,托盘不在位信号为False,检测结束标志位为False,检测结束标志位为True,则不检测,检测结束标志位为False,则检测
                count=0#托盘到位信号为True,托盘不在位信号为False,检测结束标志位为False,检测结束标志位为True,则不检测,检测结束标志位为False,则检测,循环计数器,每5次循环,检测一次托盘是否到位,如果到位,则开始检测,如果不到位,则不检测,如果检测到空框,则重置为0,重新开始检测,如果检测到有物体,则重置为0,重新开始检测,如果检测到托盘不在位,则重置为0,重新开始检测,如果检测到托盘到位,则重置为0,重新开始检测,
                print("新托盘首次到位")
                logging.error("new tray in position")
            tray_in_posold = tray_in_pos
            timer = threading.Timer(0.1, run)  
            timer.start()  

    except Exception as e:#如果读取opcua信号失败,则打印异常,并将opcua连接标志位设置为False,如果opcua连接标志位为False,则不发送opcua信号,如果opcua连接标志位为True,则发送opcua信号
        print("opc 读取信号异常")
        logging.error("run error log." + str(e))
        opc_con_flag = False
        tray_flag = True
        # loop_end = False
    finally:
        a = 1
def log_config():

    LOG_FORMAT = "[%(asctime)s][%(levelname)s]: %(message)s" 
    fname = time.strftime("news_%Y%m%d.log", time.localtime())
    level = logging.ERROR
    logging.basicConfig(level=level, format=LOG_FORMAT)

    log_file_handler = TimedRotatingFileHandler(filename=fname, when="D", interval=1, backupCount=3)
    formatter = logging.Formatter(LOG_FORMAT)
    log_file_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(log_file_handler)

def get_parser():
    parser = argparse.ArgumentParser(description="Demo of argparse") 
    parser.add_argument('--opcflag', default=True, type=bool)
    parser.add_argument('--ipcamera', default=True, type=bool)

    return parser


# def test4(frame_lwpCV, i, conff, j, output,saveflag): 
def test4(frame_lwpCV, saveflag): 

    # with open("output.txt", 'w') as ff: 
    #     ff.seek(0) 
    #     ff.truncate() 
    # last_time = datetime.datetime.now()  # 获取当前时间,并赋值给last_time,用于计算时间差,如果时间差大于0,则开始检测,如果时间差小于0,则不检测,时间差为0,则不检测
    # flag = ""#检测结果,如果检测到物体,则flag为stuff,如果检测到空框,则flag为empty
    # global flag1 #检测结果,如果检测到物体,则flag1为True,如果检测到空框,则flag1为False
    # cur_time = datetime.datetime.now() 
    # if (cur_time - last_time).seconds >= 0: 
    #     if i == 1 and j == 1:
    #         flag = "stuff" 
    #         print(flag)
    #     else:                 
    #         flag = "empty"
    #         print(flag)

    nowTime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") # 记录当前时间,并转换为字符串格式,用于保存图片,图片名称为当前时间,图片格式为jpg,图片保存路径为./pic/当前日期/当前时间.jpg
    # logging.error("pic is : " + str(nowTime) + " yolo result is : " + str(conff) + " " + str(i) + ", classify result is : " + str(output[0].tolist()) + " " + str(j)) #打印日志,日志格式为:pic is : 当前时间 yolo result is : yolo检测到物体的置信度 yolo检测到物体的类别, classify result is : 分类检测到物体的置信度 分类检测到物体的类别
    nowDay = datetime.date.today()
    dir = '/media/c2f8ffc8-6760-4d38-bbcf-42361c0324d4/pic/'+str(nowDay) #图片保存路径为./pic/当前日期/当前时间.jpg,如果当前日期文件夹不存在,则创建当前日期文件夹
    if os.path.exists(dir): 
        print("cunzai") 
    else:
        os.makedirs(dir)
    if saveflag:#如果检测到物体,则保存图片,图片名称为当前时间,图片格式为jpg,图片保存路径为./pic/当前日期/当前时间.jpg
        cv2.imwrite(dir + '/' + nowTime + '.jpg', frame_lwpCV) 
    # flag1 = False #检测结果,如果检测到物体,则flag1为True,如果检测到空框,则flag1为False
    # if tray_in_pos: #如果托盘到位,则打印托盘到位,如果托盘不在位,则打印托盘不在位
    #     flag=flag+" Y" 
    # else:
    #     flag=flag+" N" 
    # cv2.putText(frame_lwpCV, flag, (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2) #在画面上打印检测结果,如果检测到物体,则flag为stuff,如果检测到空框,则flag为empty,如果托盘到位,则flag为stuff Y,如果托盘不在位,则flag为stuff N,如果托盘到位,则flag为empty Y,如果托盘不在位,则flag为empty N
    cv2.imshow("frame", frame_lwpCV) #显示画面
    cv2.waitKey(1) 


class VideoScreenshot(object): 
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src) 
        self.screenshot_interval = 0.1 
        self.frame_width = int(self.capture.get(3)) 
        self.frame_height = int(self.capture.get(4)) 
        self.startflag=False 
        self.thread = Thread(target=self.update, args=()) 
        self.thread.daemon = True 
        self.thread.start() 

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read() 
                self.startflag=True 
            else:
                self.startflag=False

    def show_frame(self):   
        if self.status: 
            cv2.imshow('frame', self.frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

    def save_frame(self):
        self.frame_count = 0
        def save_frame_thread():
            while True:
                try:
                    nowTime = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) # 记录当前时间
                    filename = '15_' + nowTime + '.jpg'
                    path = "./dataset/15"
                    if not os.path.exists(path):
                        os.makedirs(path)
                    folder = os.path.join(path, filename)
                    print(type(self.frame))
                    cv2.imwrite(folder, self.frame)
                    self.frame_count += 1
                    time.sleep(60)
                except AttributeError:
                    pass
        Thread(target=save_frame_thread, args=()).start()


if __name__ == '__main__':
    with open('para.json') as f:    #读取参数文件
        paralist=json.load(f) #转换为字典格式
    log_config()  #日志配置
    parser = get_parser()    #参数解析
    args = parser.parse_args()#参数解析
    opc_flag = args.opcflag  #opc标志位,默认为True,如果为False,则不连接opcua服务器
    ipcamera = args.ipcamera  #摄像头标志位,默认为True,如果为False,则不连接摄像头
    global client  #opcua客户端
    tray_in_pos=False #托盘到位信号,默认为False,托盘到位信号为True,托盘不在位信号为False
    opc_con_flag=False #opcua连接标志位,默认为False,如果opcua连接成功,则为True
    client = Client(paralist['OPCAdd'], timeout=10)    #opcua客户端,连接opcua服务器,超时时间为10s,如果超时则抛出异常
    
    if opc_flag:
        while 1:   
            try:
                print("now connect to opcua server") 
                client.connect()    #   连接opcua服务器,如果连接失败,则抛出异常
                opc_con_flag=True   #opcua连接成功,opcua连接标志位为True
                print("connect to opcua succ")
                break
            except Exception as e:      
                print("the connection to opcua is disconnect")
                logging.error("connect to opcua server error log." + np.str(e)) 
                time.sleep(5)       

        t1 = threading.Timer(0.1, function=run)  #定时器,每0.1s执行一次run函数,run函数用于读取opcua信号,如果opcua信号发生变化,则修改托盘到位信号
        t1.start()  #启动定时器
        node2 = client.get_node('ns=2;s=Application.OPCUA.ETRS_ETR_L_EM_R.FROM_ETRS_TrayEmpty') #获取托盘到位信号节点,如果托盘到位信号发生变化,则修改托盘到位信号,托盘到位信号为True,托盘不在位信号为False
    if ipcamera: #如果ipcamera为True,则连接摄像头
        rtsp_stream_link=paralist['CamAdd']   #获取摄像头rtsp流地址 ,rtsp_stream_link为摄像头rtsp流地址
    else:
        rtsp_stream_link= 0 
    print("now connect to camera")
    video_stream_widget = VideoScreenshot(rtsp_stream_link) #连接摄像头,如果连接失败,则抛出异常
    while 1:
        if video_stream_widget.startflag: #如果摄像头连接成功,则打印连接成功,并跳出循环
            print("connect to camera succ")
            break
        
    c = 1 #循环计数器,每5次循环,检测一次托盘是否到位,如果到位,则开始检测,如果不到位,则不检测
    count=0 #循环计数器,每5次循环,检测一次托盘是否到位,如果到位,则开始检测,如果不到位,则不检测,如果检测到空框,则重置为0,重新开始检测,如果检测到有物体,则重置为0,重新开始检测,如果检测到托盘不在位,则重置为0,重新开始检测,如果检测到托盘到位,则重置为0,重新开始检测,
    while True:
        close = signal.signal(signal.SIGINT, handle_int)  # 捕获ctrl+c信号,如果捕获到ctrl+c信号,则关闭摄像头,关闭opcua连接,并退出程序
        if video_stream_widget.startflag==False:#如果摄像头连接断开,则重连摄像头
            logging.error("error log，摄像头连接已断开，开始重连")
            camera_con_flag=False   
            while not camera_con_flag:#如果摄像头连接成功,则打印连接成功,并跳出循环,如果摄像头连接失败,则打印重连中,并继续重连
                video_stream_widget = VideoScreenshot(rtsp_stream_link) 
                for i in range(500):
                    if video_stream_widget.startflag:
                        camera_con_flag=True
                        print("摄像头重连完成")
                        break
                    else:
                        print("重连中")
        
        # if tray_flag and not loop_end: #如果托盘到位,则开始检测,如果托盘不在位,则不检测,如果检测到空框,则重置为0,重新开始检测,如果检测到有物体,则重置为0,重新开始检测,如果检测到托盘不在位,则重置为0,重新开始检测,如果检测到托盘到位,则重置为0,重新开始检测,
        if tray_flag :  
            count=count+1 
            frame_lwpCV = video_stream_widget.frame #获取摄像头画面,并裁剪,裁剪后的画面为托盘画面,托盘画面为托盘的上方,托盘的左侧,托盘的右侧,托盘的下方,托盘的四个边界
            frame_lwpCV = frame_lwpCV[157: 841, 453:1524] 
            # cur_time_begin = time.time() 
            # first_model_flag, conff = api_run(frame_lwpCV)   #调用yolov5模型,检测托盘上是否有物体,如果有物体,则first_model_flag为True,如果没有物体,则first_model_flag为False,conff为检测到物体的置信度
            # PIL_img = Image.fromarray(frame_lwpCV) 
            # sec_model_flag, output = predict_carema(PIL_img) 
            # cur_time_end = time.time()
            # print(cur_time_end - cur_time_begin) 
            if count==5: #每5次循环,检测一次托盘是否到位,如果到位,则开始检测,如果不到位,则不检测,如果检测到空框,则重置为0,重新开始检测,如果检测到有物体,则重置为0,重新开始检测,如果检测到托盘不在位,则重置为0,重新开始检测,如果检测到托盘到位,则重置为0,重新开始检测,
                test4(frame_lwpCV ,True) 
                count=0
            else:
                test4(frame_lwpCV,False)
            # empty_box = True#检测结果,如果检测到物体,则empty_box为False,如果检测到空框,则empty_box为True
        
            # if sec_model_flag == True:  #如果检测到物体,则empty_box为False,如果检测到空框,则empty_box为True
            #     empty_box = False
            # if empty_box:#如果检测到空框,则重置为0,重新开始检测,如果检测到有物体,则重置为0,重新开始检测,如果检测到托盘不在位,则重置为0,重新开始检测,如果检测到托盘到位,则重置为0,重新开始检测,
            #     loop_end=True
            #     count=0
            #     print("检测到空框，本轮检测结束")
            #     logging.error("empty det,this loop end")
            if opc_flag: #如果opcua连接成功,则发送opcua信号,如果opcua连接失败,则不发送opcua信号
                if not opc_con_flag:  #如果opcua连接断开,则重连opcua
                    while 1:
                        try:
                            print("now reconnect to opcua server")
                            client = Client(paralist['OPCAdd'], timeout=10)   
                            client.connect()   
                            opc_con_flag = True 
                            print("reconnect to opcua succ")    
                            break
                        except Exception as e:
                            print("the reconnection to opcua is fail")
                            logging.error("reconnect to opcua error log." + np.str(e))
                            time.sleep(10)
                    t1 = threading.Timer(0.1, function=run)  
                    t1.start()  
                    

                #     if opc_con_flag: #如果opcua连接成功,则发送opcua信号,如果opcua连接失败,则不发送opcua信号
                #         dv = ua.DataValue(ua.Variant(empty_box, ua.VariantType.Boolean)) #发送opcua信号,如果检测到物体,则empty_box为False,如果检测到空框,则empty_box为True
                #         node2.set_value(dv) #发送opcua信号,如果检测到物体,则empty_box为False,如果检测到空框,则empty_box为True
                #         if empty_box: 
                #             print('################the box is empty,send opc empty flag ')
                #         else:
                #             print('################the box is stuff,send opc stuff flag ')
                # except Exception as e:#如果发送opcua信号失败,则打印异常,并将opcua连接标志位设置为False,如果opcua连接标志位为False,则不发送opcua信号,如果opcua连接标志位为True,则发送opcua信号
                #     print(e)
                #     logging.error("send  opc is empty flag error log." + np.str(e))
                #     opc_con_flag = False
                # finally:
                #     a = 1
        else:#如果托盘不在位,则不检测,如果检测到空框,则重置为0,重新开始检测,如果检测到有物体,则重置为0,重新开始检测,如果检测到托盘不在位,则重置为0,重新开始检测,如果检测到托盘到位,则重置为0,重新开始检测,
            frame_lwpCV = video_stream_widget.frame #获取摄像头画面,并裁剪,裁剪后的画面为托盘画面,托盘画面为托盘的上方,托盘的左侧,托盘的右侧,托盘的下方,托盘的四个边界
            frame_lwpCV = frame_lwpCV[157: 841, 453:1524] 
            cv2.imshow("frame", frame_lwpCV) 
            cv2.waitKey(1) 
    cap.release()

# %%
