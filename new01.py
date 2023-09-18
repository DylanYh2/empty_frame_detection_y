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
sys.path.append("yolov5/") #相对路径或绝对路径
from detect import api_run
gc.collect()
device = torch.device('cuda')
infer_dst_path = 'news/'
# In[16]:
# print(infer_imgs_path)
# for infer_img_path in infer_imgs_path[:20]:
url = "rtsp://admin:v1234567@192.168.0.110:554/"
# cap = cv2.VideoCapture(0)  # 开启摄像头
#model = pdx.load_model('output/mask_rcnn_r50_fpn')

first_model_flag = False
sec_model_flag = False
flag1 = False
flag = False
tray_in_pos=False
tray_in_posold=False
tray_flag=False
opc_con_flag=False
loop_end=False
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


def run():  # 定义方法
    global tray_in_pos  # 托盘到位信号
    global opc_con_flag     
    global tray_flag    #托盘到位标志位
    global tray_in_posold   # 托盘到位信号
    global loop_end
    global count
    try:
        if opc_con_flag:
            node = client.get_node('ns=2;s=Application.OPCUA.ETRS_ETR_L_EM_R.General_TO_ETRS_PLC_LifeBeat')  # 心跳包，取115的值，写到119,100ms周期
            node1 = client.get_node('ns=2;s=Application.OPCUA.ETRS_ETR_L_EM_R.General_FROM_ETRS_ETRS_LifeBeat')
            # node2 = client.get_node('ns=2;s=Application.OPCUA.ETRS_ETR_L_EM_R.TO_ETRS_TrayInPos')
            node2 = client.get_node('ns=2;s=Application.OPCUA.HLC_TPS_L_EM_RFID_R.TO_HLC_CameraStartReq')

            # node = client.get_node('ns=2;s=通道 1.设备 1.test')  # 心跳包，取115的值，写到119,100ms周期
            # node1 = client.get_node('ns=2;s=通道 1.设备 1.test1')

            dv = ua.DataValue(ua.Variant(node.get_value(), ua.VariantType.Boolean)) 
            node1.set_value(dv) #心跳包，取115的值，写到119,100ms周期
            tray_in_pos = node2.get_value() #读取托盘到位信号
            if tray_in_pos and not tray_in_posold:  # 托盘到位信号从0-1,表示托盘首次到位，开始判断是否空框，直到检测到空框，即loop_end为True，则停止本次循环
                tray_flag = True
                loop_end = False
                count=0
                print("新托盘首次到位")
                logging.error("new tray in position")
            tray_in_posold = tray_in_pos
            # print("副线程托盘到位信号："+str(tray_in_pos))
            timer = threading.Timer(0.1, run)  # 创建定时器，每0.1秒运行一次run
            timer.start()  # 开始执行线程

    except Exception as e:
        print("opc 读取信号异常")
        logging.error("run error log." + str(e))
        opc_con_flag = False
        tray_flag = True
        loop_end = False
    finally:
        a = 1
def log_config():

    LOG_FORMAT = "[%(asctime)s][%(levelname)s]: %(message)s" # 配置输出日志格式
    fname = time.strftime("news_%Y%m%d.log", time.localtime())# 配置输出日志文件名
    level = logging.ERROR # 配置输出日志等级
    logging.basicConfig(level=level, format=LOG_FORMAT)

    # 创建TimedRotatingFileHandler对象,每天生成一个文件
    log_file_handler = TimedRotatingFileHandler(filename=fname, when="D", interval=1, backupCount=3)
    # 设置日志打印格式,
    formatter = logging.Formatter(LOG_FORMAT)
    log_file_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(log_file_handler)

def get_parser():
    parser = argparse.ArgumentParser(description="Demo of argparse") # 创建解析器
    parser.add_argument('--opcflag', default=True, type=bool)#现场需要改成True，与PLC通信
    parser.add_argument('--ipcamera', default=True, type=bool)#现场需要改成True，使用ipc视频流，而不是读取本地mp4视频

    return parser

#TODO：参数需要修改
def test4(frame_lwpCV, i, conff, j, output,saveflag): #显示图片，打印结果，保存图片
    with open("output.txt", 'w') as ff: #清空output.txt文件
        ff.seek(0) # 清空文件 
        ff.truncate() # 清空文件    
    last_time = datetime.datetime.now()  # 记录当前时间
    flag = "" #结果标志位
    global flag1 #是否有东西标志位
    cur_time = datetime.datetime.now() #记录当前时间
    if (cur_time - last_time).seconds >= 0: #如果时间间隔大于0秒
        #TODO:yolo网络和二分网络都判断有东西取消掉
        if i == 1 and j == 1: #如果yolo网络和二分网络都判断有东西
            flag = "stuff" 
            print(flag)
        else:                 
            flag = "empty"
            print(flag)

    nowTime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") # 记录当前时间
    
    #TODO:这里保存的yolo result和classify result 不报错
    logging.error("pic is : " + str(nowTime) + " yolo result is : " + str(conff) + " " + str(i) + ", classify result is : " + str(output[0].tolist()) + " " + str(j)) #记录日志
    nowDay = datetime.date.today() # 记录当前日期
    dir = '/media/c2f8ffc8-6760-4d38-bbcf-42361c0324d4/pic/'+str(nowDay) #保存路径
    if os.path.exists(dir): #如果路径存在
        print("cunzai") 
    else:
        os.makedirs(dir)
    if saveflag:
        cv2.imwrite(dir + '/' + nowTime + '.jpg', frame_lwpCV) #保存图片
    last_time = cur_time #更新时间
    flag1 = False #重置标志位
    if tray_in_pos: #如果托盘到位
        flag=flag+" Y" #结果标志位加上托盘到位标志位
    else:
        flag=flag+" N" #结果标志位加上托盘到位标志位
    cv2.putText(frame_lwpCV, flag, (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2) #在图片上打印结果
    cv2.imshow("frame", frame_lwpCV) #显示图片
    cv2.waitKey(1) #等待1ms


class VideoScreenshot(object): #视频流类
    def __init__(self, src=0):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src) #视频流地址

        # Take screenshot every x seconds
        self.screenshot_interval = 0.1 #截图间隔

        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3)) #视频流宽度
        self.frame_height = int(self.capture.get(4)) #视频流高度
        self.startflag=False #视频流是否连接成功标志位
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=()) #创建线程
        self.thread.daemon = True #设置为守护线程
        self.thread.start() #开始线程
        # self.frame = self.capture


    def update(self): #更新视频流
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():#如果视频流打开成功
                (self.status, self.frame) = self.capture.read() #读取视频流
                self.startflag=True #视频流连接成功标志位
            else:
                self.startflag=False

    def show_frame(self):   #显示视频流
        # Display frames in main program
        if self.status: #如果视频流打开成功
            cv2.imshow('frame', self.frame)

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

    def save_frame(self):
        # Save obtained frame periodically
        self.frame_count = 0
        def save_frame_thread():
            while True:
                try:
                    nowTime = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) # 记录当前时间
                    filename = '15_' + nowTime + '.jpg'
                    # basepath = 'C:\\Users\\11988\\Desktop\\empty_test\\UI\\picture_background'#背景保存路径
                    path = "./dataset/15"
                    if not os.path.exists(path):
                        os.makedirs(path)
                    folder = os.path.join(path, filename)
                    # print(folder)
                    print(type(self.frame))
                    cv2.imwrite(folder, self.frame)
                    #cv2.imwrite('frame_{}.png'.format(self.frame_count), self.frame)
                    self.frame_count += 1
                    time.sleep(60)
                except AttributeError:
                    pass
        Thread(target=save_frame_thread, args=()).start()

# t4 = threading.Thread(target=test4, name=test4)
# t4.start()
# t4.join()

if __name__ == '__main__':
    with open('para.json') as f:
        paralist=json.load(f)

    log_config() #日志配置
    parser = get_parser()   #获取参数
    args = parser.parse_args() #解析参数
    opc_flag = args.opcflag #是否与opc通信
    ipcamera = args.ipcamera #是否使用ipc视频流
    global client #opcua客户端

    #global tray_in_pos#箱子到位信号
    #global opc_con_flag#opc连接标志位
    tray_in_pos=False#托盘到位标志位

    opc_con_flag=False #opc连接标志位   
    client = Client(paralist['OPCAdd'], timeout=10)     #创建opcua客户端,通信的超时时间为 10 秒
    #client = Client('opc.tcp://10.40.13.107:4855/', timeout=10)
    #client.set_user('VILAS')
    #    client.set_password('Welcome@2021')  # 需要在连接前调用
    
    if opc_flag:
        while 1:   
            try:
                print("now connect to opcua server")
                client.connect()    #尝试连接opcua服务器
                opc_con_flag=True #opc连接标志位，连接成功为True
                print("connect to opcua succ")
                break
            except Exception as e:      #如果连接失败，打印错误信息，等待5秒后重连
                print("the connection to opcua is disconnect")
                logging.error("connect to opcua server error log." + np.str(e)) #记录错误日志
                time.sleep(5)       #等待5秒后重连

        #client.connect()
        t1 = threading.Timer(0.1, function=run)  # 创建定时器，每0.1秒运行一次run
        t1.start()  # 开始执行线程
        node2 = client.get_node('ns=2;s=Application.OPCUA.ETRS_ETR_L_EM_R.FROM_ETRS_TrayEmpty')
    if ipcamera:#如果使用ipc视频流
        #rtsp_stream_link = 'rtsp://admin:v1234567@10.40.13.110:554/cam/realmonitor?channel=1&subtype=0'
        rtsp_stream_link=paralist['CamAdd']     #rtsp视频流地址
    else:
        rtsp_stream_link= 0 #使用本地mp4视频
    # while 1:
    #     try:
    #         print("now connect to camera")
    #         video_stream_widget = VideoScreenshot(rtsp_stream_link)
    #         if video_stream_widget.startflag:
    #             print("connect to camera succ")
    #             break
    #         else:
    #             time.sleep(5)
    #     except Exception as e:
    #         print("the connection to camera fail")
    #         logging.error("error log." + np.str(e))
    #         time.sleep(5)
    print("now connect to camera")
    # camera_con_flag = False
    # while not camera_con_flag:
    #     video_stream_widget = VideoScreenshot(rtsp_stream_link)
    #     for i in range(500):
    #         if video_stream_widget.startflag:
    #             camera_con_flag = True
    #             print("connect to camera succ")
    #             break
    #         else:
    #             print("connecting")
    video_stream_widget = VideoScreenshot(rtsp_stream_link)
    #video_stream_widget.save_frame()
    while 1:
        if video_stream_widget.startflag:
            print("connect to camera succ")
            break
        
    c = 1
    count=0
    while True:
        close = signal.signal(signal.SIGINT, handle_int) # 重写ctrl+c的信号处理函数
        if video_stream_widget.startflag==False:#摄像头断开重连
            print("摄像头连接已断开，开始重连")
            logging.error("error log，摄像头连接已断开，开始重连")
            camera_con_flag=False   
            while not camera_con_flag:
                video_stream_widget = VideoScreenshot(rtsp_stream_link) 
                for i in range(500):
                    if video_stream_widget.startflag:
                        camera_con_flag=True
                        print("摄像头重连完成")
                        break
                    else:
                        print("重连中")
        #print("主线程托盘到位信号：" + str(tray_in_pos))
        if tray_flag and not loop_end: #如果托盘到位标志位为True，且本次loop未结束

            # 读取视频流
            # frame_lwpCV = cv2.imread('test/'+ str(c) + '.jpg')
            # c = c+1
            # if c > 10:
            # c = 10
            count=count+1 #计数器，用于判断是否空框
            frame_lwpCV = video_stream_widget.frame #读取视频流
            # start1 = time.perf_counter()
            frame_lwpCV = frame_lwpCV[157: 841, 453:1524] #裁剪图片
            cur_time_begin = time.time() #记录当前时间
            #################################
            first_model_flag, conff = api_run(frame_lwpCV)  # yolo网络  
            #################################
            PIL_img = Image.fromarray(frame_lwpCV) #转换成PIL格式
            sec_model_flag, output = predict_carema(PIL_img)  # 二分网络
            # print('############')
            # print(output)
            # print('############')
            #################################
            cur_time_end = time.time()
            print(cur_time_end - cur_time_begin) #打印运行时间
            if count==5: #如果计数器为5，则开始判断是否空框
                test4(frame_lwpCV, sec_model_flag, conff, sec_model_flag, output,True)  # 显示图片，打印结果
            else:
                test4(frame_lwpCV, sec_model_flag, conff, sec_model_flag, output,False)
            empty_box = True
            #TODO:这里需要修改
            if sec_model_flag == True:  # True表示有东西，即empty=False
                empty_box = False
            if empty_box:#如果为空框，则本次loop结束
                loop_end=True
                count=0
                print("检测到空框，本轮检测结束")
                logging.error("empty det,this loop end")
            if opc_flag: # 如果与opc通信
                if not opc_con_flag:  # 如果连接标志位为False，则重连
                    while 1:
                        try:
                            print("now reconnect to opcua server")
                            client = Client(paralist['OPCAdd'], timeout=10)    #创建opcua客户端,通信的超时时间为 10 秒
                            client.connect()   #尝试连接opcua服务器
                            opc_con_flag = True # opc连接标志位，连接成功为True
                            print("reconnect to opcua succ")    
                            break
                        except Exception as e:
                            print("the reconnection to opcua is fail")
                            logging.error("reconnect to opcua error log." + np.str(e))
                            time.sleep(10)
                    t1 = threading.Timer(0.1, function=run)  # 创建定时器
                    t1.start()  # 开始执行线程
                try:

                    # node2 = client.get_node('ns=4;i=131')  # 发送给plc是否空箱
                    if opc_con_flag: # 如果opc连接成功
                        dv = ua.DataValue(ua.Variant(empty_box, ua.VariantType.Boolean)) # 将是否空箱写入opc
                        node2.set_value(dv) # 将是否空箱写入opc
                        if empty_box: # 如果为空箱
                            print('################the box is empty,send opc empty flag ')
                        else:
                            print('################the box is stuff,send opc stuff flag ')
                except Exception as e:
                    print(e)
                    logging.error("send  opc is empty flag error log." + np.str(e))
                    opc_con_flag = False
                    #client.close_session()
                finally:
                    a = 1
        else:
            frame_lwpCV = video_stream_widget.frame #读取视频流
            # start1 = time.perf_counter()
            frame_lwpCV = frame_lwpCV[157: 841, 453:1524] #裁剪图片
            cv2.imshow("frame", frame_lwpCV) #显示图片
            cv2.waitKey(1) #等待1ms
            # print(first_model_flag, sec_model_flag)
    cap.release()

# %%
