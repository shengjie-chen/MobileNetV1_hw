这个库实现了使用花分类数据集实现MobileNet神经网络的模型、训练、量化代码

## MobileNet最基本模型：
*model_v1.py* MobileNet的最基本模型文件
*train.py* 与之相对应的训练文件，只训练最后一层全连接层
使用*mobilenet_sgd_68.848.pth.tar*作为预训练权重
权重输出为*MobileNetV1.pth*

***
## MobileNet简单量化权重的模型
*model_v1_Q.py*MobileNet简单量化权重的模型文件
*train_Q.py*与之相对应的训练文件，只训练最后一层全连接层
使用*mobilenet_sgd_68.848.pth.tar*作为预训练权重
标准权重输出为*MobileNetV1.pth*，量化权重输出为*MobilenetV1_Quant.pth*

***
## MobileNet略微改动并实现全面量化的模型
*mymodel_3_plus.py*MobileNet略微改动并实现全面量化的模型文件，主要相对于原模型把平均池化改为量化平均池化，并进行激活。使用*quant_ultra.py*作为子函数库
*train_Q_v2_3_plus.py*与之相对应的训练文件，训练全部网络
若无*MobileNetV1_Quant_v3_plus.pth*则使用*mobilenet_sgd_68.848.pth.tar*作为预训练权重，若有则使用*MobileNetV1_Quant_v3_plus.pth*
权重输出为*MobileNetV1_Quant_v3_plus.pth*
最新权重是*MobileNetV1_Quant_v3_plus_94.230.pth*

***
## 量化导出结构配置以及权重文件
1.运行*torch_export.py*。使用*mymodel_3_plus.py*的网络结构，以及*MobileNetV1_Quant_v3_plus_94.230.pth*权重文件
    把网络的每一层配置写出来，保存于*config.json*
    把模型里每一层的参数都变成数组，保存于*mobilenet_4w4a.npz*
2.运行*mobilenet_param_gen.py*。在脚本中设置网络加速的相关参数。其主要的子函数库为*qnn_param_reader.py*,*qnn_mem_process.py*,*quantization.py*
    传入*config.json*及*mobilenet_4w4a.npz*的数据
    导出文件在“param/hls/”中
    参数在*param.h*中，网络结构配置在*config.h*中