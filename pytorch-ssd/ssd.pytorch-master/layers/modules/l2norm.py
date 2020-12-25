import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.init as init
# 实现L2正则化，
class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        # L2正则化的直观解释https://blog.csdn.net/red_stone1/article/details/80755144
        """
        dim (int)缩减的维度，dim=0是对0维度上的一个向量求范数，返回结果数量等于其列的个数，也就是说有多少个0维度的向量， 将得到多少个范数。dim=1同理。
        keepdim（bool）保持输出的维度 。当keepdim = False时，输出比输入少一个维度（就是指定的dim求范数的维度）。而keepdim = True时，输出与输入维度相同，仅仅是输出在求范数的维度上元素个数变为1。这也是为什么有时我们把参数中的dim称为缩减的维度，因为norm运算之后，此维度或者消失或者元素个数变为1。
        """
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
