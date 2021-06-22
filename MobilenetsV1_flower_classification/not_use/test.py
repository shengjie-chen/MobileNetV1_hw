
import torch
from torch.nn import functional as F
 
"""手动定义卷积核(weight)和偏置"""
w = torch.rand(1, 1, 3, 3)  # 16种3通道的5乘5卷积核
b = torch.rand(1)  # 和卷积核种类数保持一致(不同通道共用一个bias)
 
"""定义输入样本"""
x = torch.randn(1, 1, 4, 4)  # 1张3通道的28乘28的图像
 
"""2D卷积得到输出"""
out = F.conv2d(x, w, b, stride=2, padding=1)  # 步长为1,外加1圈padding,即上下左右各补了1圈的0,
print(out.shape)
 
# out = F.conv2d(x, w, b, stride=2, padding=2)  # 步长为2,外加2圈padding
# print(out.shape)
# out = F.conv2d(x, w)  # 步长为1,默认不padding, 不够的舍弃，所以对于28*28的图片来说，算完之后变成了24*24
# print(out.shape)
print(out[0,0,0,1])
predict = b
for i in range(1,3):
    for j in range(3):
        predict += w[0,0,i,j] * x[0,0,i-1,j+1]

print(predict)
