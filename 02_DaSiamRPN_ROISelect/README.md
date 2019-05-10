# DaSiamRPN_ROISelect

在这个库中，我们主要将原始的DaSiamRPN算法进行了简单的修改，使其能够更加易用。

你可以在窗口中使用鼠标划出ROI区域，并增加了一个面向行人跟踪的新场景

:high_brightness:**强烈推荐**：顶级会议单目标跟踪领域相关文献总结 [[github](https://github.com/foolwood/benchmark_results)]

## 机器配置

- CPU：Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz
- GPU：NVIDIA GTX1070
- python 3.6
- pytorch == 1.0.0
- numpy
- opencv

库中每一个算法中均注明了python版本以及需要安装的附加包，你可以下面的代码安装需要安装的附加包

```
pip install xxx
```

## 预训练模型下载

:zap:  原文给出的三个模型均在Google Drive中，你也可以在百度网盘中下载 [[SiamRPN_Model](https://pan.baidu.com/s/1WY6_cdjR2_I_36LjfE7Rwg)] 提取码：me52

## 使用步骤

可以按照以下步骤使用本代码

```
git clone https://github.com/NklausMikealson/SOT_Code_Comment.git
cd 02_DaSiamRPN_ROISelect
python demo.py
```

## 引用

非常感谢DaSiamRPN和SiamRPN的作者，如果你认为这项代码对你的工作有所帮助，请在你的文章中引用上述两项研究。

```
@inproceedings{Zhu_2018_ECCV,
  title={Distractor-aware Siamese Networks for Visual Object Tracking},
  author={Zhu, Zheng and Wang, Qiang and Bo, Li and Wu, Wei and Yan, Junjie and Hu, Weiming},
  booktitle={European Conference on Computer Vision},
  year={2018}
}

@InProceedings{Li_2018_CVPR,
  title = {High Performance Visual Tracking With Siamese Region Proposal Network},
  author = {Li, Bo and Yan, Junjie and Wu, Wei and Zhu, Zheng and Hu, Xiaolin},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2018}
}
```

