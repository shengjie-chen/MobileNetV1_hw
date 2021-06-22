这个库实现了使用花分类数据集实现MobileNetV1_mini神经网络的硬件加速器的源代码

## 权重及网络配置参数
硬件设计的网络结构配置参数在***config.h***文件中  
网络所需的权重数据在***param.h***文件中  

## 网络硬件代码
主体在***mobilenet_top.cpp***文件中  
所使用的子函数主要在***top_funtion.hpp***中，包括数据格式调整相关函数、各种卷积模块、全连接层、量化平均层等  
一部分在***funtion.h***中，包括padding及批量归一化的函数
最后一部分在***sliding_window_unit.h***中，包括实现华创模块的函数

## Test Bench代码
主体***mobilenet_tb.cpp***中，子函数在***tb_funtion.h***中
