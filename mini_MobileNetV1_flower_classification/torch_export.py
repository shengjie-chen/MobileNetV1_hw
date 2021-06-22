import torch    
import numpy as np
import json
import mymodel_mini

# 小型量化网络
model = mymodel_mini.MobileNetV1Qua(num_classes=5)
# model = mymodel.TempNetQua()
# print(model)
state = torch.load('MobileNetV1_Quant_mini_90.1.pth', map_location='cpu')
model.load_state_dict(state)
# model.load_state_dict(torch.load('model.pkl', map_location='cpu'))

def generate_config(model, in_shape):
    """把网络的每一层配置写出来"""
    feature_map_shape = in_shape
    print(in_shape)
    dic = {}
    conv_cnt = 0
    pool_cnt = 0
    linear_cnt = 0
    # cnt = 0
    for sub_module in model.modules():
        if type(sub_module).__base__ is torch.nn.Conv2d:
            conv_cur = {}
            conv_cur['in_shape'] = feature_map_shape[:]         
            feature_map_shape[0] = sub_module.out_channels
            feature_map_shape[1] = (feature_map_shape[1] + 2 * sub_module.padding[0] - sub_module.kernel_size[0]) // sub_module.stride[0] + 1
            feature_map_shape[2] = (feature_map_shape[2] + 2 * sub_module.padding[0] - sub_module.kernel_size[0]) // sub_module.stride[0] + 1
            conv_cur['out_shape'] = feature_map_shape[:]
            conv_cur['k'] = sub_module.kernel_size[0]
            conv_cur['s'] = sub_module.stride[0]
            conv_cur['p'] = sub_module.padding[0]
            
            dic['conv_' + str(conv_cnt)] = conv_cur
            
            conv_cnt = conv_cnt + 1
            # cnt = cnt + 1

        elif type(sub_module) is torch.nn.AvgPool2d:
            pool_cur = {}
            pool_cur['in_shape'] = feature_map_shape[:]
            pool_cur['p'] =  sub_module.kernel_size

            feature_map_shape[1] = feature_map_shape[1] // sub_module.kernel_size
            feature_map_shape[2] = feature_map_shape[2] // sub_module.kernel_size

            pool_cur['out_shape'] = feature_map_shape[:]

            dic['pool_' + str(pool_cnt)] = pool_cur

            pool_cnt = pool_cnt + 1
            # cnt = cnt + 1
        elif type(sub_module).__base__ is torch.nn.Linear:
            linear_cur = {}
            linear_cur['in_len'] = sub_module.in_features
            linear_cur['out_len'] = sub_module.out_features

            dic['linear_' + str(linear_cnt)] = linear_cur
            linear_cnt = linear_cnt + 1
            # cnt = cnt + 1
    
    return dic
    

def generate_params(model):
    """把模型里每一层的参数都变成数组"""
    dic = {}
    #print(type(dic))
    cnt = 0
    for sub_module in model.modules():
        # print(sub_module)
        if type(sub_module).__base__ is torch.nn.Conv2d:
            w = sub_module.weight.detach().numpy()
            dic['arr_' + str(cnt)] = w
            cnt = cnt + 1
            if sub_module.bias is not None:
                w = sub_module.bias.detach().numpy()
                dic['arr_' + str(cnt)] = w
                cnt = cnt + 1
        elif type(sub_module).__base__ is torch.nn.Linear:
            w = sub_module.weight.detach().numpy()
            dic['arr_' + str(cnt)] = w
            cnt = cnt + 1
        elif type(sub_module) is torch.nn.BatchNorm2d or type(sub_module) is torch.nn.BatchNorm1d:
            gamma = sub_module.weight.detach().numpy()
            dic['arr_' + str(cnt)] = gamma
            cnt = cnt + 1
            beta = sub_module.bias.detach().numpy()
            dic['arr_' + str(cnt)] = beta
            cnt = cnt + 1
            mean = sub_module.running_mean.numpy()
            dic['arr_' + str(cnt)] = mean
            cnt = cnt + 1
            var = sub_module.running_var.numpy()
            dic['arr_' + str(cnt)] = var
            cnt = cnt + 1
            eps = sub_module.eps
            dic['arr_' + str(cnt)] = eps
            cnt = cnt + 1
    return dic

dic = generate_params(model)
np.savez('mobilenet_4w4a.npz', **dic)

# dic = generate_config(model, [3, 416, 416])
dic = generate_config(model, [3, 224, 224])
print(dic)

json_str = json.dumps(dic, indent=4)
with open('config.json' , 'w') as json_file:
    json_file.write(json_str)


            