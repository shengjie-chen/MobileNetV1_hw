import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from quant_ultra import *

from mymodel_3_plus import MobileNetV1Qua

#import torchvision.models.mobilenet
import warnings

warnings.filterwarnings("ignore")
W_BIT = 4
A_BIT = 4

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    batch_size = 16
    epochs = 15

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224), #随机长宽比裁剪
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()]),
                                    #  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor()])}
                                    #  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # create model
    net = MobileNetV1Qua()
    # ck = {}
    # load pretrain weights
    model_load_path = './MobileNetV1_Quant_v3_plus.pth'
    if os.path.exists(model_load_path):
        checkpoint = torch.load(model_load_path, map_location=device)
        print("use self-pretrain weight")
        
    else:
##############
        model_weight_path = "./mobilenet_sgd_68.848.pth.tar"
    ##############
        assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
        print("use other's pretrain")
        checkpoint = torch.load(model_weight_path, map_location=device)
        #pre_weights = torch.load(model_weight_path, map_location=device)
        for k in list(checkpoint['state_dict']):
            a = k[7:]
            checkpoint['state_dict'][a] = checkpoint['state_dict'].pop(k)
        # for k in list(checkpoint['state_dict']):
        #     a = k
        #     if a.startswith('model'):
        #         if a[8] == str(3):
        #             a1=list(a)
        #             a1[8] = str(4)
        #             a=''.join(a1)
        #         elif a[8] == str(4):
        #             a1=list(a)
        #             a1[8] = str(5)
        #             a=''.join(a1)
        #         elif a[9] == str(3):
        #             a1=list(a)
        #             a1[9] = str(4)
        #             a=''.join(a1)
        #         elif a[9] == str(4):
        #             a1=list(a)
        #             a1[9] = str(5)
        #             a=''.join(a1)
        #     ck[a] = checkpoint['state_dict'][k]
            
        #checkpoint['state_dict'] = 
        net.load_state_dict(checkpoint['state_dict'])

    # delete classifier weights
    in_channel = net.fc.in_features
    linear_q = linear_Q_fn(W_BIT)
    net.fc = linear_q(in_channel, 5)
    if os.path.exists(model_load_path):
        net.load_state_dict(checkpoint)
    # freeze features weights
    # for param in net.parameters():
    #     param.requires_grad = False
    # for param in net.fc.parameters():
    #     param.requires_grad = True
    
    # no freeze
    for param in net.parameters():
        param.requires_grad = True
    

    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    best_acc = 0.0
    save_path = './MobileNetV1_Quant_v3_plus.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            #images = images.float()
            #avgin = net.get_avgin(images.to(device))
            #avgout = net.get_avgout(images.to(device))
            #aqout = net.get_aqout(images.to(device))
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print("best val accurate is " + str(best_acc))
    print("save param in " + save_path)
    print('Finished Training')


if __name__ == '__main__':
    main()
