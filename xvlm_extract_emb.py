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

from models.xvlm import XVLMBase

config = {
    "use_clip_vit": False,
    "use_swin": True,
    "vision_config": 'configs/config_swinB_224.json',
    "image_res": 224,
    "patch_size": 32,
    "use_roberta": False,
    "text_config": 'configs/config_bert.json',
    "text_encoder": 'data/bert-base-uncased',
    "embed_dim": 256,
    "temp": 0.07
}

class ImageEmbeddingsExtractor:
    def __init__(self, model):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def extract(self, image_tensor):
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            image_embeds, _ = self.model.get_vision_embeds(image_tensor)
        return image_embeds

# Instantiate the XVLMBase model
xvlm_model = XVLMBase(config)

# Create an instance of the embeddings extractor
embed_extractor = ImageEmbeddingsExtractor(xvlm_model)


for dataset in ['single-object', 'two-object', 'rel']:
    for split in ['train', 'val','gen']:

        if dataset == 'rel':
            data = RelDataset(split=split)
        else:
            data = ObjectDataset(split=split, dataset=dataset)

        dataloader = DataLoader(data, batch_size=32, shuffle=False)

        embeddings = []
        for images, _ in dataloader:
            features = embed_extractor.extract(images)
            embeddings.append(features.cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)

        embeddings_avg = embeddings.mean(axis=1)
        linear_resize = torch.nn.Linear(1024, 768)
        embeddings_avg_torch = torch.tensor(embeddings_avg)
        embeddings_resized = linear_resize(embeddings_avg_torch)

        np.save('embeddings/xvlm_{0}_{1}_embeddings.npy'.format(split), embeddings_resized.detach().numpy())