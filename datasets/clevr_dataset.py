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
import seaborn as sns
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)
from PIL import Image
import clip

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

DATASET_DIR = "data/datasets/"

DATASET_PATHS = {
    "single-object": {
        "train_image_path": DATASET_DIR + "single_object/images/",
        "val_image_path": DATASET_DIR + "single_object/images/",
        "gen_image_path": DATASET_DIR + "single_object/images/",
        "train_label_path": DATASET_DIR + "single_object/train.csv",
        "val_label_path": DATASET_DIR + "single_object/val.csv",
        "gen_label_path": DATASET_DIR + "single_object/test.csv",
    },
    "rel": {
        "train_image_path": DATASET_DIR + "rel/images/train/",
        "val_image_path": DATASET_DIR + "rel/images/val/",
        "gen_image_path": DATASET_DIR + "rel/images/gen/",
        "train_label_path": DATASET_DIR + "rel/train.json",
        "val_label_path": DATASET_DIR + "rel/val.json",
        "gen_label_path": DATASET_DIR + "rel/gen.json",
    },
    "two-object": {
        "train_image_path": DATASET_DIR + "two_object/images/train/",
        "val_image_path": DATASET_DIR + "two_object/images/val/",
        "gen_image_path": DATASET_DIR + "two_object/images/gen/",
        "train_label_path": DATASET_DIR + "two_object/train.csv",
        "val_label_path": DATASET_DIR + "two_object/val.csv",
        "gen_label_path": DATASET_DIR + "two_object/gen.csv",
    },
}

BICUBIC = InterpolationMode.BICUBIC
n_px = 224

preprocess = Compose(
    [
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)

REL_PARAMS = {
    "relations": ["front", "behind", "left", "right"],
    "rel-opposites": {
        "front": "behind",
        "behind": "front",
        "left": "right",
        "right": "left",
    },
    "nouns": ["cube", "sphere", "cylinder"],
}

class ObjectDataset(Dataset):
    def __init__(self, split, dataset):
        self.split = split
        self.dataset = dataset
        # img_dir is the directory where the actual images are stored
        self.img_dir = DATASET_PATHS[dataset][f"{split}_image_path"]

        # df is a pandas dataframe that contains the labels for each image
        self.df = pd.read_csv(DATASET_PATHS[dataset][f"{self.split}_label_path"])
        # label_mapping = {}
    
        # for ind, val in enumerate(self.df['pos'].unique().tolist()):
        #     label_mapping[val] = ind
        # self.df['label'] = self.df['pos'].apply(lambda x: label_mapping[x])
        # self.df.drop(['pos', 'neg_0', 'neg_1', 'neg_2', 'neg_3'], axis=1, inplace=True)
        
        self.attributes = [
            "blue",
            "brown",
            "cyan",
            "gray",
            "green",
            "purple",
            "red",
            "yellow",
        ]
        self.nouns = ["cube", "sphere", "cylinder"]

        self.concepts = self.attributes + self.nouns
        self.concept_to_idx = dict(
            [(concept, i) for i, concept in enumerate(self.concepts)]
        )

    def __len__(self):
        # the length of the dataset is the total number of positive labels (differs per image)
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.img_dir + self.df.iloc[idx]["file_name"]
        texts = self.df.iloc[idx][["pos", "neg_0", "neg_1", "neg_2", "neg_3"]].tolist()
        # texts = self.df.iloc[idx][["label"]].tolist()

        image = Image.open(img_path)  # Image from PIL module
        # transform image to tensor so we can return it
        image = preprocess(image)
#         label = 0

        # returns image tensor, possible captions, label of correct caption
        return image, texts

class RelDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.img_dir = DATASET_PATHS["rel"][f"{split}_image_path"]

        # load the labels from the json file
        label_file = DATASET_PATHS["rel"][f"{split}_label_path"]
        with open(label_file, "r") as l:
            self.labels = json.load(l)

        self.ims_labels = [
            (im, p) for im in self.labels for p in self.labels[im]["pos"]
        ]

        self.rel_opposites = REL_PARAMS["rel-opposites"]
        self.nouns = REL_PARAMS["nouns"]
        self.objects = REL_PARAMS["nouns"]
        self.relations = REL_PARAMS["relations"]

        self.concepts = self.objects + self.relations
        self.concept_to_idx = dict(
            [(concept, i) for i, concept in enumerate(self.concepts)]
        )

    def __len__(self):
        return len(self.ims_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir + self.ims_labels[idx][0]
        image = Image.open(img_path)  # Image from PIL module
        image = preprocess(image)

        subj, rel, obj = self.ims_labels[idx][1].strip().split()
        distractors = []
        distractors.append(f"{obj} {rel} {subj}")
        distractors.append(f"{subj} {self.rel_opposites[rel]} {obj}")
        other_nouns = list(set(self.nouns).difference(set([subj, obj])))
        assert len(other_nouns) == 1
        other_noun = other_nouns[0]

        # other_noun = random.choice(other_nouns)
        distractors.append(f"{other_noun} {rel} {obj}")
        distractors.append(f"{subj} {rel} {other_noun}")
        texts = [self.ims_labels[idx][1]] + distractors

        # shuffle the texts and return the label of the correct text
        indices = list(range(len(texts)))
        random.shuffle(indices)
        texts = [texts[i] for i in indices]

        return image, texts