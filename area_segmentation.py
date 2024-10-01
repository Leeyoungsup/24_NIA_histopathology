import matplotlib.pyplot as plt
import numpy as np
import helper
import time
import datetime
import torch.nn as nn
import torchvision.models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torchvision.utils
import torch
import pandas as pd
from torchinfo import summary
from PIL import Image
from torchvision.transforms import ToTensor
from glob import glob
from torch.utils.data import Dataset, DataLoader, random_split
from copy import copy
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from sklearn.metrics import classification_report
from tqdm import tqdm
import math
from torcheval.metrics import BinaryAccuracy
import os
import timm
import segmentation_models_pytorch as smp
import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from timm import create_model
import cv2
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
batch_size = 4
img_size = 1024
class_list = ['NT_stroma', 'NT_epithelial',
              'NT_immune',
              'Tumor',]
tf = ToTensor()
topilimage = torchvision.transforms.ToPILImage()


def createDirectory(directory):
    """_summary_
        create Directory
    Args:
        directory (string): file_path
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


img_path = '../../data/area_segmentation/_STNT/image/'
img_list = glob(img_path+'*.jpeg')
mask_list = [i.replace('/image', '/mask/npy') for i in img_list]
mask_list = [i.replace('.jpeg', '.npy') for i in mask_list]
train_img_list, test_img_list, train_mask_list, test_mask_list = train_test_split(
    img_list, mask_list, test_size=0.2, random_state=42)

test_image = torch.zeros((len(test_img_list), 3, img_size, img_size))
test_mask = torch.zeros((len(test_img_list), len(
    class_list), img_size, img_size), dtype=torch.float32)
train_image = torch.zeros((len(train_img_list), 3, img_size, img_size))
train_mask = torch.zeros((len(train_img_list), len(
    class_list), img_size, img_size), dtype=torch.float32)
for i in tqdm(range(len(train_img_list))):
    train_image[i] = tf(Image.open(train_img_list[i]))
    np_mask = tf(np.load(train_mask_list[i])/255)
    train_mask[i] = np_mask
for i in tqdm(range(len(test_img_list))):
    test_image[i] = tf(Image.open(test_img_list[i]))
    np_mask = tf(np.load(test_mask_list[i])/255)
    test_mask[i] = np_mask


class CustomDataset(Dataset):
    def __init__(self, image_list, label_list):
        self.img_path = image_list
        self.label = label_list

    def trans(self, image, label):
        if random.random() > 0.5:
            transform = transforms.RandomHorizontalFlip(1)
            label = transform(label)
            image = transform(image)

        if random.random() > 0.5:
            transform = transforms.RandomVerticalFlip(1)
            label = transform(label)
            image = transform(image)

        return image, label

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        image_path, label_path = self.trans(
            self.img_path[idx], self.label[idx])

        return image_path, label_path


train_dataset = CustomDataset(train_image, train_mask)

test_dataset = CustomDataset(test_image, test_mask)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

model = smp.UnetPlusPlus(
    # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_name="efficientnet-b7",
    # use `imagenet` pre-trained weights for encoder initialization
    encoder_weights="imagenet",
    # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    in_channels=3,
    # model output channels (number of classes in your dataset)
    classes=len(class_list),
).to(device)


def dice_loss(pred, target, num_classes=len(class_list)):
    smooth = 1e-6
    dice_per_class = torch.zeros((len(pred), num_classes)).to(pred.device)
    pred = F.softmax(pred, dim=1)
    for i in range(len(pred)):
        for class_id in range(num_classes):
            pred_class = pred[i, class_id, ...]
            target_class = target[i, class_id, ...]

            intersection = torch.sum(pred_class * target_class)
            A_sum = torch.sum(pred_class * pred_class)
            B_sum = torch.sum(target_class * target_class)
            dice_per_class[i, class_id] = (
                2. * intersection + smooth) / (A_sum + B_sum + smooth)

    return 1-dice_per_class.mean()


# summary(model, (batch_size, 3, img_size, img_size))

train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

MIN_loss = 5000
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
metrics = defaultdict(float)
for epoch in range(1000):
    train = tqdm(train_dataloader)
    count = 0
    running_loss = 0.0
    acc_loss = 0
    for x, y in train:
        model.train()
        y = y.to(device).float()
        count += 1
        x = x.to(device).float()
        optimizer.zero_grad()  # optimizer zero 로 초기화
        predict = model(x).to(device)
        cost = dice_loss(predict, y)  # cost 구함
        acc = 1-cost.item()
        cost.backward()  # cost에 대한 backward 구함
        optimizer.step()
        running_loss += cost.item()
        acc_loss += acc
        y = y.to('cpu')

        x = x.to('cpu')
        train.set_description(
            f"epoch: {epoch+1}/{1000} Step: {count+1} dice_loss : {running_loss/count:.4f} dice_score: {1-running_loss/count:.4f}")
    train_loss_list.append((running_loss/count))
    train_acc_list.append((acc_loss/count))
# test
    val = tqdm(test_dataloader)
    model.eval()
    count = 0
    val_running_loss = 0.0
    acc_loss = 0
    with torch.no_grad():
        for x, y in val:
            y = y.to(device).float()
            count += 1
            x = x.to(device).float()

            predict = model(x).to(device)
            cost = dice_loss(predict, y)  # cost 구함
            acc = 1-cost.item()
            val_running_loss += cost.item()
            acc_loss += acc
            y = y.to('cpu')
            x = x.to('cpu')
            val.set_description(
                f"test epoch: {epoch+1}/{1000} Step: {count+1} dice_loss : {val_running_loss/count:.4f}  dice_score: {1-val_running_loss/count:.4f}")
        val_loss_list.append((val_running_loss/count))
        val_acc_list.append((acc_loss/count))

    if MIN_loss > (val_running_loss/count):
        createDirectory('../../model/synth_autolabel/_STNT/')
        torch.save(model.state_dict(),
                   '../../model/synth_autolabel/_STNT/check.pt')
        MIN_loss = (val_running_loss/count)
    torch.save(model.state_dict(),
               '../../model/synth_autolabel/_STNT/'+str(epoch)+'.pt')
    pred_mask1 = torch.argmax(predict[0], 0).cpu()
    pred_mask = torch.zeros((3, img_size, img_size))
    pred_mask[0] += torch.where(pred_mask1 == 0, 1, 0)
    pred_mask[1] += torch.where(pred_mask1 == 1, 1, 0)
    pred_mask[2] += torch.where(pred_mask1 == 2, 1, 0)
    pred_mask[0] += torch.where(pred_mask1 == 3, 1, 0)
    pred_mask[1] += torch.where(pred_mask1 == 3, 1, 0)
    label_mask1 = torch.argmax(y[0], 0).cpu()
    label_mask = torch.zeros((3, img_size, img_size))
    label_mask[0] += torch.where(label_mask1 == 0, 1, 0)
    label_mask[1] += torch.where(label_mask1 == 1, 1, 0)
    label_mask[2] += torch.where(label_mask1 == 2, 1, 0)
    label_mask[0] += torch.where(label_mask1 == 3, 1, 0)
    label_mask[1] += torch.where(label_mask1 == 3, 1, 0)
    label_overlay = x[0].cpu()*0.7+label_mask*0.3
    pred_overlay = x[0].cpu()*0.7+pred_mask*0.3
    createDirectory('../../result/synth_autolabel/_STNT/')
    topilimage(torch.concat((label_overlay, pred_overlay), 2)).save(
        '../../result/synth_autolabel/_STNT/'+str(epoch)+'.jpeg')
torch.save(model.state_dict(), '../../model/synth_autolabel/_STNT/final.pt')
