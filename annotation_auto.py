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
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
import cv2
import xml.etree.ElementTree as ET
from scipy.ndimage import gaussian_filter
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
batch_size = 1
img_size = 1024

carcinoma = 'STIN'
class_list = ['NT_stroma', 'NT_epithelial',
              'NT_immune',
              'Tumor',]
tf = ToTensor()


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


def binary_mask_to_polygon(binary_mask):
    # binary_mask는 2차원 numpy array여야 합니다.
    # Contours를 찾습니다.
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= 1000]
    polygons = []
    for contour in contours:
        # 각 contour를 polygon으로 변환
        if len(contour) >= 3:  # 유효한 polygon을 만들기 위해서 최소한 3개의 점이 필요합니다.
            poly = Polygon(shell=[(point[0][0], point[0][1])
                           for point in contour])
            polygons.append(poly)

    if len(polygons) > 1:
        # 여러 개의 polygon이 있을 경우 MultiPolygon으로 변환
        return MultiPolygon(polygons)
    elif len(polygons) == 1:
        return MultiPolygon(polygons)
    else:
        return None


def mask2polygon(mask):
    NT_stroma_poly = binary_mask_to_polygon(mask[..., 0])

    NT_epithelial_poly = binary_mask_to_polygon(mask[..., 1])
    NT_immune_poly = binary_mask_to_polygon(mask[..., 2])
    Tumor_poly = binary_mask_to_polygon(mask[..., 3])

    NT_epithelial_polygon_arrays = []
    NT_immune_polygon_arrays = []
    NT_stroma_polygon_arrays = []
    Tumor_polygon_arrays = []

    if Tumor_poly != None:
        for polygon in Tumor_poly.geoms:
            if polygon.area > 5000:
                exterior_coords = np.array(polygon.exterior.coords)
                Tumor_polygon_arrays.append(exterior_coords)

    if NT_stroma_poly != None:
        for polygon in NT_stroma_poly.geoms:
            exterior_coords = np.array(polygon.exterior.coords)
            NT_stroma_polygon_arrays.append(exterior_coords)

    if NT_immune_poly != None:
        for polygon in NT_immune_poly.geoms:
            if polygon.area > 5000:  # 면적이 min_area보다 큰 경우에만 추가
                exterior_coords = np.array(polygon.exterior.coords)
                NT_immune_polygon_arrays.append(exterior_coords)

    if NT_epithelial_poly != None:
        for polygon in NT_epithelial_poly.geoms:
            if polygon.area > 5000:
                exterior_coords = np.array(polygon.exterior.coords)
                NT_epithelial_polygon_arrays.append(exterior_coords)

    return NT_stroma_polygon_arrays, NT_epithelial_polygon_arrays, NT_immune_polygon_arrays, Tumor_polygon_arrays


def polygon2asap(label_polygon, class_list, save_path):
    # 루트 엘리먼트 생성
    root = ET.Element("ASAP_Annotations")
    # Annotations 엘리먼트 생성 및 루트에 추가
    annotations = ET.SubElement(root, "Annotations")
    for i in range(len(label_polygon)):

        for j in range(len(label_polygon[i])):
            annotation = ET.SubElement(
                annotations, "Annotation", Name=class_list[i], Type="Polygon", PartOfGroup="None", Color="#F4FA58")
            coordinates = ET.SubElement(annotation, "Coordinates")
            for k in range(len(label_polygon[i][j])):
                ET.SubElement(coordinates, "Coordinate", Order=str(k), X=str(
                    float(label_polygon[i][j][k, 0])), Y=str(float(label_polygon[i][j][k, 1])))

    tree = ET.ElementTree(root)
    tree.write(save_path)


def smooth_multiclass_mask(mask, sigma=2):
    """
    다중 클래스 세그멘테이션 마스크를 부드럽게 만드는 함수입니다.

    Parameters:
        mask (np.ndarray): softmax를 적용한 다중 클래스 세그멘테이션 마스크.
                           shape은 (H, W, num_classes)이어야 합니다.
        sigma (float): Gaussian 블러의 표준 편차. 값이 클수록 마스크가 더 부드럽게 됩니다.

    Returns:
        np.ndarray: 부드럽게 처리된 softmax 마스크.
    """
    # 각 클래스 채널에 대해 Gaussian 블러 적용
    smooth_mask = np.zeros_like(mask)
    for i in range(mask.shape[-1]):
        smooth_mask[:, :, i] = gaussian_filter(mask[:, :, i], sigma=sigma)

    # 각 픽셀에 대해 softmax 재적용
    smooth_mask = np.exp(smooth_mask) / \
        np.sum(np.exp(smooth_mask), axis=-1, keepdims=True)

    return smooth_mask


def polygon2mask(image_shape, NT_stroma_polygons, NT_epithelial_polygons, NT_immune_polygons, Tumor_polygons):
    # 빈 마스크 생성 (모든 채널을 0으로 초기화)
    NT_epithelial_mask = np.zeros(
        (image_shape[0], image_shape[1]), dtype=np.uint8)
    NT_immune_mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    NT_stroma_mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    Tumor_mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    mask = np.zeros((image_shape[0], image_shape[1], 4), dtype=np.uint8)

    # 각 다각형 리스트를 순회하면서 마스크의 해당 채널에 채우기
    for polygon in NT_epithelial_polygons:
        polygon = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(NT_epithelial_mask, [polygon], 255)

    for polygon in NT_immune_polygons:
        polygon = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(NT_immune_mask, [polygon], 255)

    for polygon in NT_stroma_polygons:
        polygon = np.array(polygon, dtype=np.int32)  # 좌표를 int32로 변환
        cv2.fillPoly(NT_stroma_mask, [polygon], 255)

    for polygon in Tumor_polygons:
        polygon = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(Tumor_mask, [polygon], 255)

    mask[..., 0] += NT_stroma_mask
    mask[..., 1] += NT_epithelial_mask
    mask[..., 2] += NT_immune_mask
    mask[..., 3] += Tumor_mask
    return mask


image_list = glob('../../result/synth/'+carcinoma+'/**/*.jpeg')
random.shuffle(image_list)
xml_path = '../../result/synth/'+carcinoma+'/'
category_list = [os.path.basename(os.path.dirname(f)) for f in image_list]


class CustomDataset(Dataset):
    def __init__(self, image_list, label_list):
        self.img_path = image_list
        self.label = label_list
        self.tf = ToTensor()

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        path = self.img_path[idx]
        image = self.tf(Image.open(
            self.img_path[idx]).resize((img_size, img_size)))
        label = self.label[idx]
        return image, label, path


dataset = CustomDataset(image_list, category_list)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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

    return 1-dice_per_class


model.load_state_dict(torch.load(
    '../../model/synth_autolabel/_'+carcinoma+'/check.pt', map_location=device))


topilimage = torchvision.transforms.ToPILImage()
class_list1 = ['NT_epithelial',
               'NT_immune',
               'Tumor',]
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
train_loss_list = []
val_loss_list = np.zeros((len(dataloader), len(class_list)))
train_acc_list = []
val_acc_list = []
MIN_loss = 5000
metrics = defaultdict(float)
model.eval()
count = 0
val_running_loss = 0.0
acc_loss = 0
with torch.no_grad():
    for x, label, path in tqdm(dataloader):
        count += 1
        label = label[0]
        path = path[0]
        x = x.to(device).float()
        predict = model(x).to(device)
        x = x.to('cpu')
        mask1 = np.zeros((img_size, img_size, 3))
        pred_softmax = predict[0].argmax(dim=0).cpu()
        one_hot = torch.zeros_like(predict[0]).cpu()
        pred_softmax = np.array(one_hot.scatter_(
            0, pred_softmax.unsqueeze(0), 1))
        mask = np.zeros((img_size, img_size, len(class_list)), dtype=np.uint8)
        mask[..., 1] = cv2.morphologyEx(pred_softmax[1], cv2.MORPH_OPEN, k)*255
        mask[..., 2] = cv2.morphologyEx(pred_softmax[2], cv2.MORPH_OPEN, k)*255
        mask[..., 3] = cv2.morphologyEx(pred_softmax[3], cv2.MORPH_OPEN, k)*255
        # mask1=smooth_multiclass_mask(mask)
        # mask[...,0]=np.where(mask1.argmax(axis=2)==0,255,0)
        # mask[...,1]=np.where(mask1.argmax(axis=2)==1,255,0)
        # mask[...,2]=np.where(mask1.argmax(axis=2)==2,255,0)
        # mask[...,3]=np.where(mask1.argmax(axis=2)==3,255,0)
        NT_stroma_polygons, NT_epithelial_polygons, NT_immune_polygons, Tumor_polygons = mask2polygon(
            mask)
        label_polygon = [NT_epithelial_polygons,
                         NT_immune_polygons, Tumor_polygons]
        save_path = xml_path+label+'/' + \
            os.path.basename(path).split('.')[0]+'.xml'
        polygon2asap(label_polygon, class_list1, save_path)
        # mask2=polygon2mask((1024,1024),NT_stroma_polygons,NT_epithelial_polygons,NT_immune_polygons,Tumor_polygons)
        # mask1[...,0]+=mask2[...,1]
        # mask1[...,1]+=mask2[...,2]
        # mask1[...,2]+=mask2[...,3]
        # image=x.squeeze().permute(1,2,0).numpy()
        # image=image*255
        # overlay=image*0.8+mask1*0.2
        # overlay=overlay.astype(np.uint8)
        # overlay=Image.fromarray(overlay)
        # createDirectory('../../result/synth_autolabel/overlay/'+carcinoma+'/'+label+'/')
        # overlay.save('../../result/synth_autolabel/overlay/'+carcinoma+'/'+label+'/'+os.path.basename(path))
        # if count==100:
        #     break
