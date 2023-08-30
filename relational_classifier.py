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
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, multilabel_confusion_matrix

def encode_rel_labels(labels, lb_noun1=None, lb_relation=None, lb_noun2=None):
    
    split_labels = [label.split() for label in labels]
    noun1_labels = [label[0] for label in split_labels]
    relation_labels = [label[1] for label in split_labels]
    noun2_labels = [label[2] for label in split_labels]
    
    
    if lb_noun1 is None:
        lb_noun1 = LabelBinarizer()
        encoded_noun1 = lb_noun1.fit_transform(noun1_labels)
    else:
        encoded_noun1 = lb_noun1.transform(noun1_labels)
        
    if lb_relation is None:
        lb_relation = LabelBinarizer()
        encoded_relation = lb_relation.fit_transform(relation_labels)
    else:
        encoded_relation = lb_relation.transform(relation_labels)
        
    if lb_noun2 is None:
        lb_noun2 = LabelBinarizer()
        encoded_noun2 = lb_noun2.fit_transform(noun2_labels)
    else:
        encoded_noun2 = lb_noun2.transform(noun2_labels)
    
    encoded_labels = np.hstack([encoded_noun1, encoded_relation, encoded_noun2])
    
    return encoded_labels, lb_noun1, lb_relation, lb_noun2

relations = ['front', 'behind', 'left', 'right']
nouns = ['cube', 'sphere', 'cylinder']

class RelationalDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings_tensor, labels_tensor):
        self.embeddings = embeddings_tensor[::4]  
        self.labels = labels_tensor

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label1 = self.labels[4 * idx]
        label2 = self.labels[4 * idx + 1]
        label3 = self.labels[4 * idx + 2]
        label4 = self.labels[4 * idx + 3]
        return embedding, (label1, label2, label3, label4)

def get_labels(split):
    df = pd.read_csv('data/datasets/rel/{0}.csv'.format(split))
    labels = df['pos'].tolist()
    return labels

train_labels = get_labels('train')
val_labels = get_labels('val')
test_labels = get_labels('gen')

encoded_train_labels, lb_noun1_train, lb_relation_train, lb_noun2_train = encode_rel_labels(train_labels)
encoded_val_labels, _, _, _ = encode_rel_labels(val_labels, lb_noun1_train, lb_relation_train, lb_noun2_train)
encoded_test_labels, _, _, _ = encode_rel_labels(test_labels, lb_noun1_train, lb_relation_train, lb_noun2_train)

train_labels_tensor = torch.FloatTensor(encoded_train_labels)
val_labels_tensor = torch.FloatTensor(encoded_val_labels)
test_labels_tensor = torch.FloatTensor(encoded_test_labels)
                                                    
def get_dataloader(model_name, split):
    emb = np.load('embeddings/{0}_rel_{1}_embeddings.npy'.format(model_name, split))
    if split == 'train':
        labels_tensor = train_labels_tensor
    elif split == 'val':
        labels_tensor = val_labels_tensor
    elif split == 'gen':
        labels_tensor = test_labels_tensor
    

    embeddings_tensor = torch.FloatTensor(emb)
    dataset = RelationalDataset(embeddings_tensor, labels_tensor)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    return dataloader, embeddings_tensor, labels_tensor

class RelationalClassifier(nn.Module):
    def __init__(self, input_dim, shared_dim):
        super(RelationalClassifier, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.7), 
            nn.BatchNorm1d(shared_dim) 
        )
        
        # Each output block consists of 3 (noun) + 4 (relation) + 3 (noun) = 10 neurons
        self.output_blocks = nn.ModuleList([
            nn.ModuleDict({
                'noun1': nn.Sequential(
                    nn.Linear(shared_dim, 3),
                    nn.Sigmoid()
                ),
                'relation': nn.Sequential(
                    nn.Linear(shared_dim, 4),
                    nn.Sigmoid()
                ),
                'noun2': nn.Sequential(
                    nn.Linear(shared_dim, 3),
                    nn.Sigmoid()
                )
            })
            for _ in range(4)
        ])
    
    def forward(self, x):
        shared = self.shared_layers(x)
        outputs = []
        for block in self.output_blocks:
            noun1_out = block['noun1'](shared)
            relation_out = block['relation'](shared)
            noun2_out = block['noun2'](shared)
            combined_out = torch.cat([noun1_out, relation_out, noun2_out], dim=1)
            outputs.append(combined_out)
        return outputs

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
    learning_rate = 0.001
    wt_decay = 1e-4
    num_epochs = 20
    
    model = RelationalClassifier(input_dim, shared_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wt_decay)

    best_val_loss = float('inf')
    no_improve_epochs = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            inputs, labels_tuple = batch
            outputs = model(inputs)
            loss = sum(criterion(output, label) for output, label in zip(outputs, labels_tuple))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_dataloader):.4f}")

        # Validation
        model.eval()
        val_loss = 0
        for batch in val_dataloader:
            inputs, labels_tuple = batch
            outputs = model(inputs)
            loss = sum(criterion(output, label) for output, label in zip(outputs, labels_tuple))
            val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= 5:
                print(f"Early stopping. Best validation loss: {best_val_loss:.4f}")
                break
    
    return model

def evaluate_model(model, data_loader):
    model.eval()
    all_true_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels_tuple = batch
            outputs = model(inputs)

            for idx, (output, true_labels) in enumerate(zip(outputs, labels_tuple)):
                pred_noun1 = (output[:, :3] > 0.5).cpu().numpy() 
                pred_relation = (output[:, 3:7] > 0.5).cpu().numpy()
                pred_noun2 = (output[:, 7:] > 0.5).cpu().numpy()

                all_preds.extend(np.hstack([pred_noun1, pred_relation, pred_noun2]))
                all_true_labels.extend(true_labels.cpu().numpy())

    all_preds = [array.astype(int) for array in all_preds]

    return all_true_labels, all_preds

def calculate_metrics(true_labels,pred_labels):
    n_images = len(true_labels) // 4
    correct_label_predictions = 0
    total_labels = 0
    total_h_loss = 0
    total_f1 = 0
    
    for i in range(0, len(true_labels), 4):
        total_labels += 4
        
        for j in range(4):
            # Check label-wise for each image for accuracy
            if any([np.array_equal(pred_labels[i+j], true_labels[i+k]) for k in range(4)]):
                correct_label_predictions += 1
            
            # Hamming loss and F1 score for the current predicted label
            hamming_distances = [np.sum(pred_labels[i+j] != true_labels[i+k]) for k in range(4)]
            best_index = np.argmin(hamming_distances)
            
            h_loss = hamming_distances[best_index] / len(pred_labels[i+j])
            f1 = f1_score(true_labels[i+best_index], pred_labels[i+j], average='micro')
            
            total_h_loss += h_loss
            total_f1 += f1
    
    average_h_loss = total_h_loss / total_labels
    average_f1 = total_f1 / total_labels
    accuracy = correct_label_predictions / total_labels
       
    return {
        "accuracy": accuracy,
        "hamming_loss": average_h_loss,
        "f1_micro": average_f1
    }

def store_metrics(model_name, metrics_dict):
    with open(f"{model_name}_new_rel_metrics.json", "w") as f:
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

with open("model_results/rel_metrics.json", "w") as f:
    json.dump(all_metrics, f)

def compute_adapted_confusion_matrices(true_labels, preds):
    num_labels = true_labels[0].shape[0]
    cms = []
    
    for label_idx in range(num_labels):
        y_true = []
        y_pred = []
        
        for i in range(0, len(true_labels), 2):
            # For the first label of the image
            if preds[i][label_idx] == 1:
                y_pred.append(1)
                if true_labels[i][label_idx] == 1 or true_labels[i+1][label_idx] == 1:
                    y_true.append(1)
                else:
                    y_true.append(0)
            else:
                y_pred.append(0)
                if true_labels[i][label_idx] == 0 and true_labels[i+1][label_idx] == 0:
                    y_true.append(0)
                else:
                    y_true.append(1)
            
            # For the second label of the image
            if preds[i+1][label_idx] == 1:
                y_pred.append(1)
                if true_labels[i+1][label_idx] == 1 or true_labels[i][label_idx] == 1:
                    y_true.append(1)
                else:
                    y_true.append(0)
            else:
                y_pred.append(0)
                if true_labels[i+1][label_idx] == 0 and true_labels[i][label_idx] == 0:
                    y_true.append(0)
                else:
                    y_true.append(1)
        
        cms.append(confusion_matrix(y_true, y_pred))
    
    return np.array(cms)


def plot_multilabel_confusion_matrices_grid(confusion_matrices):
    n_labels = confusion_matrices.shape[0]
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
    label_names = ['cube', 'sphere', 'cylinder', 'front', 'behind', 'left', 'right', 'cube', 'sphere', 'cylinder']
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
    plt.savefig('model_results/rel_cm.jpg', format='jpg', dpi=300, bbox_inches='tight')


mcm = compute_adapted_confusion_matrices(true_labs, pred_labs)
plot_multilabel_confusion_matrices_grid(mcm)
