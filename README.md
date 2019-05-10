# **SOT_Code_Comment**

在这个库中，我们主要对目前流行的单目标跟踪算法（**S**ingle **O**bject **T**racking）进行单行注释，希望这可以帮助刚刚接触这些算法的学者可以更快的理解代码的含义，相关的算法的原始代码都可以在下文的链接中下载使用。

:high_brightness:**强烈推荐**：顶级会议单目标跟踪领域相关文献总结 [[github](https://github.com/foolwood/benchmark_results)]

## 机器配置

- CPU：Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz
- GPU：NVIDIA GTX1070

库中每一个算法中均注明了python版本以及需要安装的附加包，你可以下面的代码安装需要安装的附加包

```
pip install xxx
```

## 原始代码及论文（与库中序号对应）

1. **DaSiamRPN**: Zheng Zhu*, Qiang Wang*, Bo Li*, Wei Wu, Junjie Yan, and Weiming Hu

   "Distractor-aware Siamese Networks for Visual Object Tracking." (ECCV 2018). [[paper](https://arxiv.org/pdf/1808.06048.pdf)] [[github](https://github.com/foolwood/DaSiamRPN)]

2. **DaSiamRPN_ROISelect**：基于DaSiamRPN算法进行了改进，修改了边界框初始设置步骤，并增加了新的跟踪场景数据，具体请见 [[github](https://github.com/NklausMikealson/SOT_Code_Comment/tree/master/02_DaSiamRPN_ROISelect)]





