# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


from utils import get_subwindow_tracking


def generate_anchor(total_stride, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4),  dtype=np.float32)
    size = total_stride * total_stride
    count = 0
    for ratio in ratios:
        # ws = int(np.sqrt(size * 1.0 / ratio))
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size / 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


# 追踪器初始化设置，用于初始化相关参数以及跟踪属性
class TrackerConfig(object):
    # These are the default hyper-params for DaSiamRPN 0.3827
    windowing = 'cosine'  # to penalize large displacements [cosine/uniform]
    # Params from the network architecture, have to be consistent with the training
    exemplar_size = 127  # input z size
    instance_size = 271  # input x size (search region)
    total_stride = 8
    score_size = (instance_size-exemplar_size)/total_stride+1
    context_amount = 0.5  # context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    # 设置锚点数量
    anchor_num = len(ratios) * len(scales)
    anchor = []
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295
    # adaptive change search region #
    adaptive = True

    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.score_size = (self.instance_size - self.exemplar_size) / self.total_stride + 1


# 根据上一帧跟踪结果，获得本帧结果
def tracker_eval(net, x_crop, target_pos, target_sz, window, scale_z, p):
    # 获得边界框位置以及置信分数
    delta, score = net(x_crop)

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()

    # 边界框回归修正
    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

    def change(r):
        return np.maximum(r, 1./r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    # 尺度修正
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
    # 比率修正
    r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    # 对所有预测结果进行修正
    # 修正比例
    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    # 置信值修正
    pscore = penalty * score

    # window float
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    # 获取最佳预测框（非极大值抑制NMS）
    best_pscore_id = np.argmax(pscore)

    # 划定最终预测结果，给出目标的位置和尺寸
    target = delta[:, best_pscore_id] / scale_z
    target_sz = target_sz / scale_z
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr

    res_x = target[0] + target_pos[0]
    res_y = target[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + target[2] * lr
    res_h = target_sz[1] * (1 - lr) + target[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])
    return target_pos, target_sz, score[best_pscore_id]


# 对网络进行初始化设置，输入参数：图像im，目标位置pos，目标尺寸sz，网络参数net
# 初始化设置中，设定了目标的初始位置
# 返回网络参数，以字典形式返回
def SiamRPN_init(im, target_pos, target_sz, net): 
    # 设置一个空的字典
    state = dict()  
    # 设置跟踪器的相关参数
    p = TrackerConfig() 
    # 将网络的参数加载至跟踪模型中
    p.update(net.cfg)   
    # 将图像的尺寸加载至state中
    state['im_h'] = im.shape[0] 
    state['im_w'] = im.shape[1]

    if p.adaptive:  # 初始化设置为True
        if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
            p.instance_size = 287  # small object big search region
        else:
            p.instance_size = 271   # 用于设置instance_size，为score_size计算做准备

        # 与TrackerConfig类中计算方式一致
        p.score_size = (p.instance_size - p.exemplar_size) / p.total_stride + 1 

    # 生成候选框尺寸锚点
    p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, int(p.score_size))   

    # 计算图像均值
    avg_chans = np.mean(im, axis=(0, 1))

    # context_amount参数已经初始化，默认0.5，
    wc_z = target_sz[0] + p.context_amount * sum(target_sz) # target_sz是模板的横纵尺寸，通过sum函数加在一起
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    # round取整数
    s_z = round(np.sqrt(wc_z * hc_z))
    # initialize the exemplar 初始化目标模板
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

    # 数据降维
    z = Variable(z_crop.unsqueeze(0))
    # 将网络加载至GPU上运行
    net.temple(z.cuda())

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    # 将所有的参数写入字典中，进行保存
    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    # 返回网络的初始化设置 以字典形式返回
    return state


# 对目标进行跟踪，输入参数： 跟踪器-state 图像-im
# 返回跟踪目标的位置target_pos 尺寸target_size 得分score
def SiamRPN_track(state, im):
    # 接收网络参数
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    # 主要接收上一帧目标跟踪的位置以及尺寸
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    # 更新搜索区域，决定本帧的搜索区域（根据上一帧的检测结果来设置本帧搜索区域）
    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    # 获取尺度变化率
    scale_z = p.exemplar_size / s_z
    # 调整搜索区域
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # extract scaled crops for search region x at previous target position
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))

    # 获得本帧预测结果，target_pos-目标位置 target_sz-目标尺度 score-置信分数
    # Ps: target_sz * scale_z表示本帧的搜索区域大小
    target_pos, target_sz, score = tracker_eval(net, x_crop.cuda(), target_pos, target_sz * scale_z, window, scale_z, p)
    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state
