from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch
"""
PriorBox的作用是与ground truth相比较，找到overlap大于阈值的框，
然后计算这些框预测的4个值与 ground truth 的smooth_l1_loss ,计算置信度loss同理。
"""
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    # feature_maps ： 特征图大小，用来遍历生成每一个点的坐标并加上0.5作为先验框的坐标中心。
    'feature_maps': [38, 19, 10, 5, 3, 1],
    # min_dim：图片大小
    'min_dim': 300,
    # steps ：代码中用于将先验框中心坐标归一化，除以图片大小，使得求出来的大小与image_size有关，与feature_map_size无关
    'steps': [8, 16, 32, 64, 100, 300],
    # min_sizes：用于计算每一个先验框的长与宽，self.min_sizes[k]/self.image_size。与论文中的默认框（先验框）的宽高与图片大小的比例有关，如下公式：Sk=Smin+((Smax-Smin)/m-1)*(k-1)
    'min_sizes': [30, 60, 111, 162, 213, 264],
    # max_sizes： 是为了对应一个特例，1′ 的宽与高比例的大小，如下公式：Sk'=（Sk*Sk+1）~1/2
    'max_sizes': [60, 111, 162, 213, 264, 315],
    # aspect_ratios：代码中已经显示的定义了[1, 1']大小的默认框，还差[2, 3, 1/2, 1/3] 宽高比的定义。针对6个特征图[38, 19 ,10 ,5 ,3 , 1],在实际上并不是每一个特征图的每个点都用了6个比例的框，而是[4, 6, 6, 6, 4, 4]。
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    #variance：变化变动，学习率
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

# 生成default box的
class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source feature map.
    """
    # cfg是什么意思？？？？下边的自变量的每个参数什么意思？464464
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        # 图片的大小
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        # # 存放的是feature map的尺寸:38,19,10,5,3,1
        for k, f in enumerate(self.feature_maps):
            # from itertools import product as product
            for i, j in product(range(f), repeat=2):
                # 这里实际上可以用最普通的for循环嵌套来代替, 主要目的是产生anchor的坐标(i,j)
                f_k = self.image_size / self.steps[k] # steps=[8,16,32,64,100,300]. f_k大约为feature map的尺寸
                # unit center x,y   # 计算中心的距离
                # 求得center的坐标, 浮点类型. 实际上, 这里也可以直接使用整数类型的 `f`, 计算上没太大差别
                # 求出来的大小与image_size有关 ，与feature_size无关
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                # 这里一定要特别注意 i,j 和cx, cy的对应关系, 因为cy对应的是行, 所以应该令cy与i对应.
                # aspect_ratios 为1时对应的box  aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # 根据原文, 当 aspect_ratios 为1时, 会有一个额外的 box, 如下:
                # rel size: sqrt(s_k * s_(k+1))
                # aspect_ratio: 1.
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                # 其余(2, 或 2,3)的宽高比(aspect ratio)
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
                # 综上, 每个卷积特征图谱上每个像素点最终产生的 box 数量要么为4, 要么为6, 根据不同情况可自行修改.
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            # clamp_ 是clamp的原地执行版本
            output.clamp_(max=1, min=0)
        return output
    # 输出default box坐标(可以理解为anchor box)
#     输出为：38*38*4+19*19*6+10*10*6+5*5*6+3*3*4+1*1*4=8732
