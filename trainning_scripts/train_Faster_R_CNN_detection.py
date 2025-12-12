import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights, FastRCNNPredictor
from Deeplearning.project3331.dataset.face_detection_dataset import  Face_detection_dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor , Resize , Compose,ToPILImage
from torchvision.transforms import Compose, Resize, RandomAffine, ColorJitter, ToTensor, Normalize


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_transform = Compose([
        Resize((416, 416)),
        # ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

    val_transform = Compose([
        Resize((416, 416)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])
    train_set=Face_detection_dataset(root='../data/face_detection',train=True,transforms=train_transform)
    val_set=Face_detection_dataset(root='../data/face_detection',train=False,transforms=val_transform)
    train_params={
        "batch_size":6,
        "shuffle":True,
        "num_workers":6,
        "drop_last":True,
    }
    val_params={
        "batch_size":6,
        "shuffle":True,
        "num_workers":6,
        "drop_last":False,
    }
    train_loader = DataLoader(train_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # print(model.roi_heads.box_predictor.cls_score.in_features)
    # print(model.roi_heads.box_predictor.cls_score.out_features)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=model.roi_heads.box_predictor.cls_score.in_features,num_classes=2)
    # print(model.roi_heads.box_predictor)

if __name__ == '__main__':
    train()