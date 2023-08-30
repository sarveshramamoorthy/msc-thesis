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
import clip

from datasets.clevr_dataset import ObjectDataset, RelDataset

# Function to get CLIP embeddings
def get_embeddings(model, dataloader):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            features = model.encode_image(images)
            embeddings.append(features.cpu().numpy())
            labels.append(batch_labels)
    embeddings = np.concatenate(embeddings)
    return embeddings, labels


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, _ = clip.load('ViT-L/14')
model.to(device)

for dataset in ['single-object', 'two-object', 'rel']:
    for split in ['train', 'val','gen']:

        if dataset == 'rel':
            data = RelDataset(split=split)
        else:
            data = ObjectDataset(split=split, dataset=dataset)

        dataloader = DataLoader(data, batch_size=32, shuffle=False)

        embeddings, labels = get_embeddings(model, dataloader)

        np.save('embeddings/clip_{0}_{1}_embeddings.npy'.format(dataset,split), embeddings.numpy())