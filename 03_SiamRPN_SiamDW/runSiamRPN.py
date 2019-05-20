import cv2
import glob
import argparse
import numpy as np
import lib.models.models as models
from lib.tracker.siamrpn import SiamRPN
from easydict import EasyDict as edict
from lib.utils.utils import load_pretrain, cxy_wh_2_rect

def parse_args():
    """
    args for rpn testing.
    """
    parser = argparse.ArgumentParser(description='PyTorch SiamRPN Tracking Test')
    parser.add_argument('--arch', dest='arch', default='SiamRPNRes22', help='backbone architecture')
    parser.add_argument('--resume', dest='resume', default='./snapshot/CIResNet22_RPN.pth', help='pretrained model')
    parser.add_argument('--anchor_nums', default=5, type=int, help='anchor numbers')
    parser.add_argument('--epoch_test', default=False, type=bool, help='multi-gpu epoch test flag')
    args = parser.parse_args()

    return args

def get_depth_info(info):
    depth_info = []
    [rows, cols, deps] = info.shape
    for i in range(rows - 1):
        for j in range(cols - 1):
            for k in range(deps - 1):
                if info[i, j, k] != 'nan' and info[i, j, k] != 0:
                    depth_info.append(info[i, j, k])

    depth_info = np.mean(depth_info)
    return depth_info

def text_save(filename, data):    # filename为写入文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')# 去除[],这两行按数据不同，可以选择
        s = s.replace("'","").replace(',','') +'\n'   # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print('Successful Saved')


# 跟踪目标
def track(tracker, net):
    # 设置起始帧、损失时间以及运行时间
    start_frame, lost_times, toc = 0, 0, 0
    rgb_image_files = sorted(glob.glob('./image/Scene3/rgb/*.png'))
    depth_image_files = sorted(glob.glob('./image/Scene3/depth_bmp/*.bmp'))
    depth_save_list = []

    # 批量处理
    for f, image_file in enumerate(rgb_image_files):

        # 设置计数时间点
        tic = cv2.getTickCount()

        # 如果是第一帧
        if f == start_frame:  # init
            # HxWxC 读取图像
            rgb_im = cv2.imread(rgb_image_files[f])
            depth_im = cv2.imread(depth_image_files[f])
            # 判别图像是否为灰度图像，如果是，那么转化成rgb进行处理
            if len(rgb_im.shape) == 2:
                rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_GRAY2BGR)
            # Select ROI
            cv2.namedWindow("SiamRPN", cv2.WND_PROP_FULLSCREEN)
            # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            try:
                init_rect = cv2.selectROI('SiamRPN', rgb_im, False, False)
                x, y, w, h = init_rect
            except:
                exit()
            cx, cy, w, h = x, y, w, h
            # 通过初始位置，获取目标位置以及目标大小
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            # 将上述位置信息传入至跟踪器初始模块，作为模板图像
            state = tracker.init(rgb_im, target_pos, target_sz, net)  # init tracker
            info = depth_im[int(target_pos[0])-2 : int(target_pos[0])+3, int(target_pos[1])-2 : int(target_pos[1])+3]
            depth_info = get_depth_info(info)
            # depth_info = np.mean(depth_im[int(target_pos[0]), int(target_pos[1])])
            print(depth_info)
            depth_save_list.append(depth_info)
            
        # 如果不是第一帧，开始跟踪
        elif f > start_frame:  # tracking
            rgb_im = cv2.imread(rgb_image_files[f])
            depth_im = cv2.imread(depth_image_files[f])
            # 判别图像是否为灰度图像，如果是，那么转化成rgb进行处理
            if len(rgb_im.shape) == 2:
                rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_GRAY2BGR)
            # 将上一帧的跟踪结果作为输入，进入跟踪器进行跟踪
            state = tracker.track(state, rgb_im)  # track
            target_pos = state['target_pos']
            info = depth_im[int(target_pos[0]) - 2: int(target_pos[0]) + 3, int(target_pos[1]) - 2: int(target_pos[1]) + 3]
            depth_info = get_depth_info(info)
            # depth_info = np.mean(depth_im[int(target_pos[0]), int(target_pos[1])])
            print(depth_info)
            depth_save_list.append(depth_info)
            # 通过预测得到的state，提取目标位置
            res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            res = [int(l) for l in res]
            # 跟踪框绘制
            cv2.rectangle(rgb_im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
            # 显示跟踪结果
            cv2.imshow('SiamRPN', rgb_im)
            # 减缓显示速度
            cv2.waitKey(1)            
        
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    return depth_save_list


def main():
    # 接收控制台输入参数
    args = parse_args()

    # prepare model
    # 读取模型、载入模型
    net = models.__dict__[args.arch](anchors_nums=args.anchor_nums)
    net = load_pretrain(net, args.resume)
    net.eval()
    # 将模型网络放入GPU中运行
    net = net.cuda()

    # prepare tracker
    # 利用easydict建立一个全局的变量，模型所有的参数都可以通过调用info属性得到，简化字典调用数据的过程
    info = edict()
    # 将控制台参数中网络模型参数、数据集、是否需要使用多GPU的标志写入全局变量中
    info.arch = args.arch
    info.epoch_test = args.epoch_test
    # 初始化跟踪器（info中的参数作为输入写进tracker中）
    tracker = SiamRPN(info)

    # 跟踪目标
    depth_save_list = track(tracker, net)
    text_save('./depth_result.txt', depth_save_list)

if __name__ == '__main__':
    main()