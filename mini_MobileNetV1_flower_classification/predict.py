import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


# from model_v2 import MobileNetV2
from mymodel_mini import MobileNetV1Qua

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         ])

    # load image
    img_path = "./rose.jpeg"
    # img_path = "./tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = MobileNetV1Qua(num_classes=5).to(device)
    # load model weights
    model_weight_path = "./MobileNetV1_Quant_mini_90.1.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        output_14 = np.array(model.get_14out(img.to(device)))
        output_15 = np.array(model.get_15out(img.to(device)))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print(output)
    
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
