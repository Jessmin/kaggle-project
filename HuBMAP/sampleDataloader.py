import torch
import PIL
from PIL import Image
import numpy as np


# 二
class DealDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, labels_path, Transform=None):
        # 1：所有图片和标签的路径
        images_path_list = []
        labels_path_list = []

        """"""
        # 在这里写，获得所有image路径，所有label路径的代码，并将路径放在分别放在images_path_list和labels_path_list中
        """"""
        self.images_path_list = images_path_list
        self.labels_path_list = labels_path_list
        self.transform = Transform

    def __getitem__(self, index):
        # 2：根据index取得相应的一幅图像，一幅标签的路径

        image_path = image_path_list[index]
        label_path = label_path_list[index]

        # 3：将图片和label读出。“L”表示灰度图，也可以填“RGB”

        image = Image.open(image_path).convert("L")
        label = Image.open(label_path).convert("L")

        # 4：tansform 参数一般为 transforms.ToTensor()，意思是上步image,label 转换为 tensor 类型

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

    def __len__(self):
        return len(self.images_path_list)

# 一
images_path = ""
labels_path = ""
dataset = DealDataset(images_path,labels_path,Transform = transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset = dataset,batch_size = 16,shuffle = False) #shuffle 填True 就会打