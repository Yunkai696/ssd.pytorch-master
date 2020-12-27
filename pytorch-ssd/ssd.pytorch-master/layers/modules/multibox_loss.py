# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp

"""
    # 计算目标:
    # 输出那些与真实框的iou大于一定阈值的框的下标.
    # 根据与真实框的偏移量输出localization目标
    # 用难样例挖掘算法去除大量负样本(默认正负样本比例为1:3)
    # 目标损失:
    # L(x,c,l,g) = (Lconf(x,c) + αLloc(x,l,g)) / N
    # 参数:
    # c: 类别置信度(class confidences)
    # l: 预测的框(predicted boxes)
    # g: 真实框(ground truth boxes)
    # N: 匹配到的框的数量(number of matched default boxes)
"""
class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes#类别数
        self.threshold = overlap_thresh#交并比阈值0.5
        self.background_label = bkg_label#背景标签0
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching#True
        self.do_neg_mining = neg_mining#True
        self.negpos_ratio = neg_pos# 负样本和正样本的比例, 3:1
        self.neg_overlap = neg_overlap# 0.5 判定负样本的阈值.
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        # loc_data: [batch_size, 8732, 4]
        # conf_data: [batch_size, 8732, 21]
        # priors: [8732, 4]  default box 对于任意的图片, 都是相同的, 因此无需带有 batch 维度
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)# num = batch_size
        priors = priors[:loc_data.size(1), :] # loc_data.size(1) = 8732, 因此 priors 维持不变
        num_priors = (priors.size(0))# num_priors = 8732
        num_classes = self.num_classes# num_classes = 21 (默认为voc数据集)

        # 将priors(default boxes)和ground truth boxes匹配
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(., num_priors, 4)  # shape:[batch_size, 8732, 4]
        conf_t = torch.LongTensor(num, num_priors)  # shape:[batch_size, 8732]
        for idx in range(num):
            # targets是列表, 列表的长度为batch_size, 列表中每个元素为一个 tensor,
            # 其 shape 为 [num_objs, 5], 其中 num_objs 为当前图片中物体的数量, 第二维前4个元素为边框坐标, 最后一个元素为类别编号(1~20)
            truths = targets[idx][:, :-1].data  # [num_objs, 4]
            labels = targets[idx][:, -1].data  # [num_objs] 使用的是 -1, 而不是 -1:, 因此, 返回的维度变少了
            defaults = priors.data  # [8732, 4]
            # from ..box_utils import match
            # 关键函数, 实现候选框与真实框之间的匹配, 注意是候选框而不是预测结果框! 这个函数实现较为复杂, 会在后面着重讲解
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t,idx)
            # 注意! 要清楚 Python 中的参数传递机制, 此处在函数内部会改变 loc_t, conf_t 的值, 关于 match 的详细讲解可以看后面的代码解析
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # 用Variable封装loc_t, 新版本的 PyTorch 无需这么做, 只需要将 requires_grad 属性设置为 True 就行了
        # pytorch中的变量都是variable的形式
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0  # 筛选出 >0 的box下标(大部分都是=0的)
        # 回归函数仅对标签大于0的坐标有意义，背景框不在预测的范围之内
        num_pos = pos.sum(dim=1, keepdim=True)  # 求和, 取得满足条件的box的数量, [batch_size, num_gt_threshold]

        # 位置(localization)损失函数, 使用 Smooth L1 函数求损失
        # loc_data:[batch, num_priors]
        # pos_idx: [batch, num_priors, 4], 复制下标成坐标格式, 以便获取坐标值
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)  # 获取预测结果值，选框位置预测
        loc_t = loc_t[pos_idx].view(-1, 4)  # 获取gt值，训练集给出的选框位置
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)  # 计算损失
        # 计算最大的置信度, 以进行难负样本挖掘
        # conf_data: [batch, num_priors, num_classes]
        # batch_conf: [batch, num_priors, num_classes]
        batch_conf = conf_data.view(-1, self.num_classes)  # reshape

        # conf_t: [batch, num_priors]rs, 4]
        #         # pos: [batch, num_priors]
        # loss_c: [batch*num_priors, 1], 计算每个priorbox预测后的损失
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # 难负样本挖掘, 按照loss进行排序, 取loss最大的负样本参与更新
        loss_c[pos.view(-1, 1)] = 0  # 将所有的pos下标的box的loss置为0(pos指示的是正样本的下标)
        # 将 loss_c 的shape 从 [batch*num_priors, 1] 转换成 [batch, num_priors]
        loss_c = loss_c.view(num, -1)  # reshape
        # 进行降序排序, 并获取到排序的下标
        _, loss_idx = loss_c.sort(1, descending=True)
        # 将下标进行升序排序, 并获取到下标的下标
        _, idx_rank = loss_idx.sort(1)
        # num_pos: [batch, 1], 统计每个样本中的obj个数
        num_pos = pos.long().sum(1, keepdim=True)
        # 根据obj的个数, 确定负样本的个数(正样本的3倍)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        # 获取到负样本的下标
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # 计算包括正样本和负样本的置信度损失
        # pos: [batch, num_priors]
        # pos_idx: [batch, num_priors, num_classes]
        # 增加一个维度，在2的位置
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        # neg: [batch, num_priors]
        # neg_idx: [batch, num_priors, num_classes]
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # 按照pos_idx和neg_idx指示的下标筛选参与计算损失的预测数据
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        # 按照pos_idx和neg_idx筛选目标数据
        targets_weighted = conf_t[(pos + neg).gt(0)]
        # 计算二者的交叉熵
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # 将损失函数归一化后返回
        N = num_pos.data.sum()
        loss_l = loss_l / N
        loss_c = loss_c / N
        return loss_l, loss_c
