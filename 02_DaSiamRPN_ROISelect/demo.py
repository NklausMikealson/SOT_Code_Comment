# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join

from net import SiamRPNvot
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect

# load net
# 初始化网络模型
net = SiamRPNvot()  
# 网络模型参数读取
net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNVOT.model')))  
# 将其放在GPU上运行
net.eval().cuda()   

# image and init box
# 设置图像目录
image_files = sorted(glob.glob('./ETH/*.jpg'))  
# HxWxC 读取图像
im = cv2.imread(image_files[0])

# Select ROI
cv2.namedWindow("SiamRPN", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
try:
    init_rect = cv2.selectROI('SiamRPN', im, False, False)
    x, y, w, h = init_rect
except:
    exit()

# tracker init
# 目标位置转化 将目标的中心位置标记为target_pos 尺度大小标记为target_sz 
target_pos = np.array([x + w / 2, y + h / 2])
target_sz = np.array([w, h])

# 初始化网络  
state = SiamRPN_init(im, target_pos, target_sz, net)    

# tracking and visualization
toc = 0
# 批量处理（伪视频流处理）
for f, image_file in enumerate(image_files):    
    # 读取图片
    im = cv2.imread(image_file) 
    tic = cv2.getTickCount()
    # track
    state = SiamRPN_track(state, im)  
    toc += cv2.getTickCount()-tic
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
    res = [int(l) for l in res]
    # 跟踪框绘制
    cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)   
    # 显示跟踪结果
    cv2.imshow('SiamRPN', im)  
    # 减缓显示速度 
    cv2.waitKey(30)

print('Tracking Speed {:.1f}fps'.format((len(image_files)-1)/(toc/cv2.getTickFrequency())))
