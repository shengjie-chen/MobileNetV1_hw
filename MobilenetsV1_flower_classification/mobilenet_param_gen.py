from qnn_param_reader import QNNParamReader
from qnn_mem_process import QNNLayerMemProcess
import numpy as np
import json
import os
import sys
# 1st
# # conv       0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26 fclayer
# w_bit   =   [4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,     4]
# in_bit  =   [8,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,     5] 
# out_bit =   [4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,     4] 
# l_shift =   [8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,     8]
# simd    =   [3,  3,  8,  3,  8,  3,  8,  3,  8,  3,  8,  3,  8,  3,  8,  3,  8,  3,  8,  3,  8,  3,  8,  3,  8,  3,  8,     8]   
# pe      =   [16, 1,  8,  1,  2,  1,  2,  1,  2,  1,  2,  1,  2,  1,  2,  1,  2,  1,  2,  1,  2,  1,  2,  1,  2,  1,  2,     5]
    
#2nd
# conv       0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26 fclayer
w_bit   =   [4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,     4]
in_bit  =   [8,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,     5] 
out_bit =   [4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,     4] 
l_shift =   [8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,     8]
simd    =   [3,  9,  8,  9,  8,  9,  8,  3,  8,  3,  8,  3,  8,  3,  8,  3,  8,  3,  8,  3,  8,  3,  8,  3,  8,  3,  8,     1]   
pe      =   [4,  1,  2,  1,  2,  1,  4,  1,  2,  1,  4,  1,  2,  1,  4,  1,  4,  1,  4,  1,  4,  1,  4,  1,  2,  1,  4,     1]

if __name__ == "__main__":

    target_dir_hls_param = 'param/hls/'
    if not os.path.exists(target_dir_hls_param):
        os.makedirs(target_dir_hls_param)
    
    hls_param_file = open(target_dir_hls_param + 'param.h', 'w')
    hls_config_file = open(target_dir_hls_param + 'config.h', 'w')

    config_file = open('config.json', 'r', encoding='utf-8')
    config = json.load(config_file)
    reader = QNNParamReader('mobilenet_4w4a.npz')

    # conv_0 - 26
    for i in range(0, 27):
        # print("######################")
        # print(i)
        processer = QNNLayerMemProcess('conv_' + str(i), reader, config, w_bit=w_bit[i], in_bit=in_bit[i], out_bit=out_bit[i], l_shift=l_shift[i], pe=pe[i], simd=simd[i])
        w, inc, bias = processer.conv() # 写成pe * tiles 以及 pe * 1 形式
        param_str = processer.layer_param_to_init_str(w, inc, bias)
        # print(param_str)
        config_str = processer.conv_config_str()
        # print(config_str)
        hls_param_file.write(param_str)
        hls_config_file.write(config_str)
    
    i = i + 1
    # fclayer 0
    processer = QNNLayerMemProcess('linear_' + str(0), reader, config, w_bit=w_bit[i], in_bit=in_bit[i], out_bit=out_bit[i], l_shift=l_shift[i], pe=pe[i], simd=simd[i])
    w = processer.last_linear()
    param_str = processer.last_layer_param_to_init_str(w)
    # print(param_str)
    config_str = processer.last_linear_config_str()
    # print(config_str)
    hls_param_file.write(param_str)
    hls_config_file.write(config_str)

    # last_bias = reader.get_last()
    # np.save('param/hls/last_bias', last_bias)
    # last_bias.tofile('param/hls/last_bias.bin')

    hls_param_file.close()
    hls_config_file.close()