import argparse
import json
import logging
import os
import pickle
import pprint
import random

import numpy as np
import torch
import tqdm
import pandas as pd
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from datasets.clevr_dataset import ObjectDataset, RelDataset

from models.blip import blip_feature_extractor

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth'
model = blip_feature_extractor(pretrained=model_url, image_size=224, vit='base')
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_blip_embeddings(model, dataloader):
    embeddings = []
    with torch.no_grad():
        for images, batch_labels in dataloader:
            batch_embeddings = []
            for idx in range(len(images)):
                image = images[idx]  
                image = image.unsqueeze(0).to(device)

                features = model(image, "", mode='image')[0, 0]
                batch_embeddings.append(features.cpu().numpy())
            embeddings.append(batch_embeddings)
    
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings

for dataset in ['single-object', 'two-object', 'rel']:
    for split in ['train', 'val','gen']:

        if dataset == 'rel':
            data = RelDataset(split=split)
        else:
            data = ObjectDataset(split=split, dataset=dataset)

        dataloader = DataLoader(data, batch_size=32, shuffle=False)

        embeddings = get_blip_embeddings(model, dataloader)

        np.save('embeddings/blip_{0}_{1}_embeddings.npy'.format(dataset,split))