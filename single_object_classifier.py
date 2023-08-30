import argparse
import json
import logging
import os
import pickle
import pprint
import random
from itertools import chain, product

import numpy as np
import torch
import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, multilabel_confusion_matrix


adjectives = ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow']
nouns = ['cube', 'sphere', 'cylinder']


def get_dataloader(model_name, split):
    emb = np.load('embeddings/{0}_single_object_{1}_embeddings.npy'.format(model_name, split))
    df = pd.read_csv('data/datasets/single_object/{0}.csv'.format(split))
    labels = df['pos'].tolist()
    label_words = [label.split() for label in labels]

    mlb = MultiLabelBinarizer(classes=adjectives + nouns)
    encoded_labels = mlb.fit_transform(label_words)

    embeddings_tensor = torch.FloatTensor(emb)
    labels_tensor = torch.FloatTensor(encoded_labels)
    dataset = TensorDataset(embeddings_tensor, labels_tensor)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    return dataloader, embeddings_tensor, labels_tensor


class SingleObjectClassifier(nn.Module):
    def __init__(self, input_dim, shared_dim, adj_dim, noun_dim):
        super(SingleObjectClassifier, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.adj_layers = nn.Sequential(
            nn.Linear(shared_dim, adj_dim),
            nn.ReLU(),
            nn.Linear(adj_dim, 8),
            nn.Sigmoid()
        )
        
        self.noun_layers = nn.Sequential(
            nn.Linear(shared_dim, noun_dim),
            nn.ReLU(),
            nn.Linear(noun_dim, 3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        shared = self.shared_layers(x)
        adj_out = self.adj_layers(shared)
        noun_out = self.noun_layers(shared)
        return adj_out, noun_out


def train_model(seed, model_name):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    train_dataloader, train_emb_tensor, train_lab_tensor = get_dataloader(model_name, 'train')
    val_dataloader, _, _ = get_dataloader(model_name, 'val')
    test_dataloader, _, _ = get_dataloader(model_name, 'gen')
    
    input_dim = 768
    shared_dim = 512
    adj_dim = 256
    noun_dim = 128
    learning_rate = 0.001
    num_epochs = 20

    model = SingleObjectClassifier(input_dim, shared_dim, adj_dim, noun_dim)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            inputs, labels = batch
            adj_labels = labels[:, :8] 
            noun_labels = labels[:, 8:]

            adj_out, noun_out = model(inputs)
            loss_adj = criterion(adj_out, adj_labels)
            loss_noun = criterion(noun_out, noun_labels)
            loss = loss_adj + loss_noun

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_dataloader):.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_dataloader:
                inputs, labels = batch
                adj_labels = labels[:, :8]
                noun_labels = labels[:, 8:]
                adj_out, noun_out = model(inputs)
                loss_adj = criterion(adj_out, adj_labels)
                loss_noun = criterion(noun_out, noun_labels)
                loss = loss_adj + loss_noun
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss / len(val_dataloader):.4f}")
    
    return model

def evaluate_model(model, data_loader):
    model.eval()
    all_adj_preds = []
    all_noun_preds = []
    all_true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            adj_out, noun_out = model(inputs)

            adj_preds = (adj_out > 0.5).float()
            noun_preds = (noun_out > 0.5).float()

            all_adj_preds.extend(adj_preds.cpu().numpy())
            all_noun_preds.extend(noun_preds.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    combined_pred = [np.concatenate((arr1, arr2)) for arr1, arr2 in zip(all_adj_preds, all_noun_preds)]

    return all_true_labels, combined_pred

def calculate_metrics(true_labels, preds):
    accuracy = accuracy_score(true_labels, preds)
    hamming = hamming_loss(true_labels, preds)
    f1_micro = f1_score(true_labels, preds, average='micro')
    
    return {
        "accuracy": accuracy,
        "hamming_loss": hamming,
        "f1_micro": f1_micro
    }

def store_metrics(model_name, metrics_dict):
    with open(f"{model_name}_1obj_metrics.json", "w") as f:
        json.dump(metrics_dict, f)

models = ["clip", "blip", "xvlm"]
seed_values = [42, 123, 456, 789, 101]

all_metrics = {}

for model_name in models:
    
    train_dataloader, train_emb_tensor, train_lab_tensor = get_dataloader(model_name, 'train')
    val_dataloader, _, _ = get_dataloader(model_name, 'val')
    test_dataloader, _, _ = get_dataloader(model_name, 'gen')
    
    model_avg_metrics = {
        "train": {"accuracy": 0, "hamming_loss": 0, "f1_micro": 0},
        "validation": {"accuracy": 0, "hamming_loss": 0, "f1_micro": 0},
        "test": {"accuracy": 0, "hamming_loss": 0, "f1_micro": 0}
    }
    
    for seed in seed_values:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        model = train_model(seed, model_name)
        
        train_true_labels, train_preds = evaluate_model(model, train_dataloader)
        val_true_labels, val_preds = evaluate_model(model, val_dataloader)
        test_true_labels, test_preds = evaluate_model(model, test_dataloader)

        train_metrics = calculate_metrics(train_true_labels, train_preds)
        val_metrics = calculate_metrics(val_true_labels, val_preds)
        test_metrics = calculate_metrics(test_true_labels, test_preds)

        model_avg_metrics["train"]["accuracy"] += train_metrics["accuracy"]
        model_avg_metrics["train"]["hamming_loss"] += train_metrics["hamming_loss"]
        model_avg_metrics["train"]["f1_micro"] += train_metrics["f1_micro"]

        model_avg_metrics["validation"]["accuracy"] += val_metrics["accuracy"]
        model_avg_metrics["validation"]["hamming_loss"] += val_metrics["hamming_loss"]
        model_avg_metrics["validation"]["f1_micro"] += val_metrics["f1_micro"]

        model_avg_metrics["test"]["accuracy"] += test_metrics["accuracy"]
        model_avg_metrics["test"]["hamming_loss"] += test_metrics["hamming_loss"]
        model_avg_metrics["test"]["f1_micro"] += test_metrics["f1_micro"]

    num_seeds = len(seed_values)
    for metric_type in ["train", "validation", "test"]:
        model_avg_metrics[metric_type]["accuracy"] /= num_seeds
        model_avg_metrics[metric_type]["hamming_loss"] /= num_seeds
        model_avg_metrics[metric_type]["f1_micro"] /= num_seeds

    all_metrics[model_name] = model_avg_metrics

    store_metrics(model_name, model_avg_metrics)

with open("model_results/single_object_metrics.json", "w") as f:
    json.dump(all_metrics, f)


def plot_multilabel_confusion_matrices_grid(confusion_matrices):
    """Plot the confusion matrices in a grid format."""
    n_labels = confusion_matrices.shape[0]
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
    label_names = ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow', 'cube', 'sphere', 'cylinder']
    for i, ax in enumerate(axes.ravel()):
        if i < n_labels:
            sns.heatmap(confusion_matrices[i], annot=True, fmt='g', cmap='Blues', ax=ax, cbar=False)
            ax.set_title(f'Label {label_names[i]}')
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')
            ax.set_xticklabels(['Negative', 'Positive'])
            ax.set_yticklabels(['Negative', 'Positive'])
            ax.set_ylim([0,2])
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig('model_results/single_object_cm.jpg', format='jpg', dpi=300, bbox_inches='tight')

mcm = multilabel_confusion_matrix(true_labs, pred_labs)

plot_multilabel_confusion_matrices_grid(mcm)