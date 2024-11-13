import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from urllib.request import urlopen
from PIL import Image
import timm
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from glob import glob
from sklearn.model_selection import train_test_split
import pytorch_model_summary as tms
import torch.nn as nn
import random
import torchmetrics
from torch.nn.modules.batchnorm import _BatchNorm
import matplotlib.pyplot as plt
import torch.nn.functional as F
print(f"GPUs used:\t{torch.cuda.device_count()}")
device = torch.device("cuda",1)
print(f"Device:\t\t{device}")
class_list=['normal','abnormal']
params={'image_size':256,
        'lr':2e-4,
        'beta1':0.5,
        'beta2':0.999,
        'batch_size':1,
        'epochs':20,
        'n_classes':2,
        'inch':3,
        }

trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

def transback(data:Tensor) -> Tensor:
    return data / 2 + 0.5
class CustomDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, params, images, labels):
        self.images = images
        self.args = params
        self.labels = labels
        
    def trans(self, image):
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
        if random.random() > 0.5:
            image = transforms.functional.vflip(image)
        return image
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = self.trans(image)
        return image, label
    
    def __len__(self):
        return len(self.images)

# DataLoader 딕셔너리를 생성하여 각 CSV 파일에 대한 DataLoader를 저장
dataloaders = {}
csv_folder = '../../data/usefulness/breast/'  # CSV 파일이 저장된 폴더 경로
csv_files = [f for f in os.listdir(csv_folder) if f.startswith('test_') and f.endswith('.csv')]

for csv_file in csv_files:
    csv_path = os.path.join(csv_folder, csv_file)
    df = pd.read_csv(csv_path)
    
    # 이미지 경로와 레이블을 리스트로 추출
    image_paths = df['image'].tolist()
    image_labels = df['label'].tolist()
    
    # 이미지 데이터를 텐서로 변환
    test_images = torch.zeros((len(image_paths), params['inch'], params['image_size'], params['image_size']))
    for i in tqdm(range(len(image_paths)), desc=f"Processing {csv_file}"):
        test_images[i] = trans(Image.open(image_paths[i]).convert('RGB').resize((params['image_size'], params['image_size'])))
    
    # 레이블을 one-hot 인코딩하여 텐서로 변환
    test_labels = torch.tensor(image_labels)
    test_labels = torch.nn.functional.one_hot(test_labels).to(torch.int64)
    
    # CustomDataset 및 DataLoader 생성
    test_dataset = CustomDataset(params, test_images, test_labels)
    dataloaders[csv_file] = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True)

# 각 DataLoader는 dataloaders 딕셔너리에서 접근 가능
# 예를 들어, test_0.csv에 대한 DataLoader는 dataloaders['test_0.csv']로 접근 가능


class FeatureExtractor(nn.Module):
    """Feature extoractor block"""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        cnn1= timm.create_model('tf_efficientnetv2_s', pretrained=True)
        self.feature_ex = nn.Sequential(*list(cnn1.children())[:-1])

    def forward(self, inputs):
        features = self.feature_ex(inputs)
        
        return features
class custom_model(nn.Module):
    def __init__(self, num_classes, image_feature_dim,feature_extractor_scale1: FeatureExtractor):
        super(custom_model, self).__init__()
        self.num_classes = num_classes
        self.image_feature_dim = image_feature_dim

        # Remove the classification head of the CNN model
        self.feature_extractor = feature_extractor_scale1
        # Classification layer
        self.classification_layer = nn.Linear(image_feature_dim, num_classes)
        
    def forward(self, inputs):
        batch_size, channels, height, width = inputs.size()
        
        # Feature extraction using the pre-trained CNN
        features = self.feature_extractor(inputs)  # Shape: (batch_size, 2048, 1, 1)
        
        # Classification layer
        logits = self.classification_layer(features)  # Shape: (batch_size, num_classes)
        
        return logits
    
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        
def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
            
import transformers

Feature_Extractor=FeatureExtractor()
raw_model = custom_model(2,1280,Feature_Extractor)
raw_model = raw_model.to(device)
source_model = custom_model(2,1280,Feature_Extractor)
source_model = source_model.to(device)
base_optimizer = torch.optim.SGD
# optimizer = SAM(model.parameters(), base_optimizer, lr=params['lr'], momentum=0.9)
raw_model.load_state_dict(torch.load('../../model/usefulness/breast/raw_usefulness_check.pt',map_location=device))
source_model.load_state_dict(torch.load('../../model/usefulness/breast/source_usefulness_check.pt',map_location=device))

# Initialize F1 score metric for binary classification
f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=2).to(device)

# Function to evaluate a model on a given DataLoader
# Function to evaluate a model on a given DataLoader
def evaluate_model_manual_f1(model, dataloader,csv_file):
    model.eval()
    
    # Initialize counts
    tp, tn, fp, fn = 0, 0, 0, 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=csv_file):
            images = images.to(device)
            labels = labels.argmax(dim=1).to(device)  # Convert one-hot encoded labels to class indices
            logits = model(images)
            preds = torch.argmax(logits, dim=1)  # Get predicted class indices
            
            # Calculate TP, TN, FP, FN
            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    
    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1_score
# Load each test set and evaluate with both models
# Dictionary to store F1 scores for each model on each test set
f1_scores = {'raw_model': {}, 'source_model': {}}

# Evaluate both raw_model and source_model on each test set DataLoader
for csv_file, dataloader in dataloaders.items():
    # Evaluate raw_model
    f1_raw = evaluate_model_manual_f1(raw_model, dataloader,'raw '+csv_file)
    f1_scores['raw_model'][csv_file] = f1_raw
    
    # Evaluate source_model
    f1_source = evaluate_model_manual_f1(source_model, dataloader,'source '+csv_file)
    f1_scores['source_model'][csv_file] = f1_source

# Convert the dictionary to a DataFrame for easier CSV export
f1_scores_df = pd.DataFrame(f1_scores)

# Save to CSV
f1_scores_df.to_csv("../../result/usefulness/breast/f1_scores.csv", index_label="Test Set")

print("F1 scores saved to f1_scores.csv")