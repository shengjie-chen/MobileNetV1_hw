This repo is the sourse code of my undergraduate graduation design.
It design a lightweight neural network MobileNet v1.
Main work:
1.It uses Pytorch to build the structure of the MobileNet network and write training scripts for training
2.It tries to use the linear quantization method to modify the network code and derive the compressed network weight. 
3.Refer to the quantization method of the UltraNet network, modify the network model and training script, and export the quantized weight file and the configuration of each layer of the network after retraining. Parameter file, some parameters of hardware implementation are also set and exported together.
4.It uses the Vivado HLS tool to write the hardware implementation code of the network. Design each sub-module circuit and write the corresponding Test Bench at the same time. After all modules are verified, connect the modules and Test Bench according to the network structure.

In the end, the whole process of the network from construction to quantification to hardware design is basically realized.