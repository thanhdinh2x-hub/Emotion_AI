


import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, List
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor , Resize , Compose,ToPILImage
import cv2
from webencodings import labels
from torchvision.transforms import Compose, Resize, RandomAffine, ColorJitter, ToTensor, Normalize

from Deeplearning.hw2 import output


class Face_detection_dataset(Dataset):
    def __init__(self, root, train=True,transforms=None):

        self.root_image = os.path.join(root, "images")
        self.root_label = os.path.join(root, "labels2")
        self.transforms = transforms # để transform ảnh
        # self.root = os.path.normpath(root)
        print(self.root_image)
        print(self.root_label)
        self.images_path = []
        self.labels_path = []
        if train:
            mode = "train"
            root_image_train = os.path.join(self.root_image,mode)
            root_label_train = self.root_label
            for image in os.listdir(root_image_train):
                if image.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.images_path.append(os.path.join(root_image_train, image))


            for label in os.listdir(root_label_train):
                if label.lower().endswith((".txt")):
                    self.labels_path.append(os.path.join(root_label_train, label))

        else:
            mode = "test"
            root_image_test = os.path.join(self.root_image, "val")
            root_label_test = self.root_label
            for image in os.listdir(root_image_test):
                if image.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.images_path.append(os.path.join(root_image_test, image))
            for label in os.listdir(root_label_test):
                if label.lower().endswith((".txt")):
                    self.labels_path.append(os.path.join(root_label_test, label))
        # print("images_path", self.images_path)
        # print("labels_path", self.labels_path)



    def __len__(self):
        return len(self.labels_path)

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        # 1) Đọc ảnh PIL
        pil_img = Image.open(image_path).convert("RGB")

        # 2) Ảnh cho OpenCV (vẽ + lưu): RGB -> numpy -> BGR
        image_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 3) Ảnh cho model (tensor)
        image = image_path
        if self.transforms:
            image = Image.open(image_path).convert("RGB")  # tạo emotion ảnh PIL và ép ảnh thành ảnh RGB có 3 chiều (bất kể là ảnh gì thì cũng bị ép)
            image = self.transforms(image)  # trả về ảnh dướng dạng tensor
        # # đọc ảnh bằng PIL
        # image_show= Image.open(image) # ko convert sang pixel đc
        # image_show.show()

        # --- đọc label2 ---
        #TODO để lấy tên file để tìm file cùng tên.txt: path = "../data/images/train/image1.jpg"
        base = os.path.basename(image_path)  # image1.jpg
        name = os.path.splitext(base)[0]  # image1
        label_name = name + ".txt"
        # print(image_path)
        # print(base)
        # print(label_name)
        label_path= os.path.join(self.root_label, label_name)
        with open(label_path, "r") as fo:
            annotations = fo.readlines()
        print(annotations)
        targets = [anno.rstrip().split(" ") for anno in annotations]
        print(targets)
        labels=[]
        bboxes=[]
        output={} # sửa lại cho mô hình fasterr CNN dùng đc data set
        for target in targets:

            # print("target:",target)
            label = " ".join(target[:2])
            labels.append(label)
            # print(label)
            xmin = int(float(target[2]))
            ymin = int(float(target[3]))
            xmax = int(float(target[4]))
            ymax = int(float(target[5]))
            bbox = [xmin, ymin, xmax, ymax]
            bboxes.append(bbox)

        # output["boxes"]=torch.FloatTensor(bboxes) # sửa lại cho mô hình fasterr CNN dùng đc data set
        # output["labels"]=torch.longTensor(labels) # sửa lại cho mô hình fasterr CNN dùng đc data set

            # cv2.rectangle(image_cv, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        # cv2.imwrite("image test.jpg", image_cv)


        return image, bboxes, labels


if __name__ == '__main__':
    transforms= Compose([
        # transforms.Resize((416, 416)), # ép ảnh lại thì nó sẽ hiện ra sample sai
        transforms.ToTensor(),
        # Normalize(mean=[0.485, 0.456, 0.406],
        #           std=[0.229, 0.224, 0.225]),
    ])

    dataset = Face_detection_dataset(root="../data/face_detection", train= True,transforms=transforms)
    # all_labels = []
    #
    # for i in range(len(dataset)):
    #     _, _, labels = dataset[i]  # labels là list của 1 ảnh
    #     all_labels.extend(labels)  # gộp vào list tổng
    #
    # num_unique = len(set(all_labels))
    # print("num unique labels in dataset:", num_unique)
    # print("unique labels:", set(all_labels))

    image, bboxes, label = dataset.__getitem__(25)
    print("image :",image)
    print("bboxes :",bboxes)
    print("label :",label)
    # image: torch.Tensor (C,H,W), [0,1]
    image_np = image.permute(1, 2, 0).cpu().numpy()  # (H,W,C), float
    image_np = (image_np * 255).astype(np.uint8)  # uint8
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    cv2.imwrite("image_detector_test.jpg", image_np)
    # length= dataset.__len__()
    # print("length of data:" ,length)
    #
    # image ,label = dataset.__getitem__(3333)
    # print("length categories :")
    # print(len(dataset.categories))
    # # # đọc ảnh bằng cv2
    # # image_show = cv2.imread(image)  # BGR
    # # cv2.imshow(str(label), image_show)
    # # cv2.waitKey(0)
    # # print("in ra:")
    # print(image)
    # print(image.shape)
    # print(image_show)
    # print(label)
    # epochs=10
    #
    # training_loader = DataLoader(
    #     dataset=dataset,
    #     batch_size=2,
    #     num_workers=4,
    #     shuffle=True, # trước khi generate ra ảnh thì tráo emotion lần đối với mẫu epochs
    #     drop_last=False,
    # )
    # print(training_loader) # chả ra cái gì ấy chả liên quan trả mỗi <torch.utils.data.dataloader.DataLoader object at 0x000002284D014AD0>
    # for epoch in range(epochs):
    #     for images,labels in training_loader: # DataLoader ko có tự trả ra image,lable mà dataset.__getitem__ để lấy ra, còn dataloader chỉ là cái vỏ đọc đẻ chia batch thôi
    #
    #         print(image.shape)
    #         print(labels)
    #         print(len(labels))
    #         to_pil = ToPILImage()
    #         for image,label in zip(images,labels):
    #             image=to_pil(image)
    #             image.show()
    #         break
    #     print('epochs thứ {}:'.format(epoch))