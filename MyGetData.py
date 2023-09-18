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
# from predict import predict_carema
from logging.handlers import TimedRotatingFileHandler
from opcua import Client, ua
import argparse
import gc
# from train import SELFMODEL
import os
import os.path as osp
import shutil
import torch.nn as nn
from PIL import Image
import sys
import json
sys.path.append("yolov5/") 
# from detect import api_run
#
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
    global tray_in_posold  
    global count
    try:
        if opc_con_flag:
            node = client.get_node('ns=2;s=Application.OPCUA.ETRS_ETR_L_EM_R.General_TO_ETRS_PLC_LifeBeat') 
            node1 = client.get_node('ns=2;s=Application.OPCUA.ETRS_ETR_L_EM_R.General_FROM_ETRS_ETRS_LifeBeat')
            node2 = client.get_node('ns=2;s=Application.OPCUA.HLC_TPS_L_EM_RFID_R.TO_HLC_CameraStartReq')
            dv = ua.DataValue(ua.Variant(node.get_value(), ua.VariantType.Boolean)) 
            node1.set_value(dv) 
            tray_in_pos = node2.get_value() 
            if tray_in_pos and not tray_in_posold: 
                tray_flag = True 
                count=0
                print("新托盘首次到位")
                logging.error("new tray in position")
            tray_in_posold = tray_in_pos
            timer = threading.Timer(0.1, run)  
            timer.start()  

    except Exception as e:
        print("opc 读取信号异常")
        logging.error("run error log." + str(e))
        opc_con_flag = False
        tray_flag = True
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


def test4(frame_lwpCV, saveflag): 
    nowTime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") 
    nowDay = datetime.date.today()
    dir = '/media/c2f8ffc8-6760-4d38-bbcf-42361c0324d4/pic/'+str(nowDay) 
    if os.path.exists(dir): 
        print("cunzai") 
    else:
        os.makedirs(dir)
    if saveflag:
        cv2.imwrite(dir + '/' + nowTime + '.jpg', frame_lwpCV) 
    cv2.imshow("frame", frame_lwpCV) 
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
    with open('para.json') as f:    
        paralist=json.load(f) 
    log_config() 
    parser = get_parser()    
    args = parser.parse_args()
    opc_flag = args.opcflag 
    ipcamera = args.ipcamera  
    global client  
    tray_in_pos=False 
    opc_con_flag=False 
    client = Client(paralist['OPCAdd'], timeout=10)   
    
    if opc_flag:
        while 1:   
            try:
                print("now connect to opcua server") 
                client.connect()  
                opc_con_flag=True   
                print("connect to opcua succ")
                break
            except Exception as e:      
                print("the connection to opcua is disconnect")
                logging.error("connect to opcua server error log." + np.str(e)) 
                time.sleep(5)       

        t1 = threading.Timer(0.1, function=run) 
        t1.start()  
        node2 = client.get_node('ns=2;s=Application.OPCUA.ETRS_ETR_L_EM_R.FROM_ETRS_TrayEmpty') #获取托盘到位信号节点,如果托盘到位信号发生变化,则修改托盘到位信号,托盘到位信号为True,托盘不在位信号为False
    if ipcamera: 
        rtsp_stream_link=paralist['CamAdd']   
    else:
        rtsp_stream_link= 0 
    print("now connect to camera")
    video_stream_widget = VideoScreenshot(rtsp_stream_link) 
    while 1:
        if video_stream_widget.startflag: 
            print("connect to camera succ")
            break
        
    c = 1 
    count=0 
    while True:
        close = signal.signal(signal.SIGINT, handle_int) 
        if video_stream_widget.startflag==False:
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
        if tray_flag :  
            count=count+1 
            frame_lwpCV = video_stream_widget.frame 
            frame_lwpCV = frame_lwpCV[157: 841, 453:1524] 
            if count==5: 
                test4(frame_lwpCV ,True) 
                count=0
            else:
                test4(frame_lwpCV,False)
            if opc_flag: 
                if not opc_con_flag:  
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
        else:
            frame_lwpCV = video_stream_widget.frame 
            frame_lwpCV = frame_lwpCV[157: 841, 453:1524] 
            cv2.imshow("frame", frame_lwpCV) 
            cv2.waitKey(1) 
    cap.release()

# %%
