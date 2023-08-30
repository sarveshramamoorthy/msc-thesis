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

class TwoObjectDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings_tensor, labels_tensor):
        self.embeddings = embeddings_tensor[::2]
        self.labels = labels_tensor

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label1 = self.labels[2 * idx]
        label2 = self.labels[2 * idx + 1]
        return embedding, (label1, label2)

def get_dataloader(model_name, split):
    emb = np.load('embeddings/{0}_two_object_{1}_embeddings.npy'.format(model_name, split))
    df = pd.read_csv('data/datasets/two_object/{0}.csv'.format(split))
    labels = df['pos'].tolist()
    label_words = [label.split() for label in labels]

    mlb = MultiLabelBinarizer(classes=adjectives + nouns)
    encoded_labels = mlb.fit_transform(label_words)

    embeddings_tensor = torch.FloatTensor(emb)
    labels_tensor = torch.FloatTensor(encoded_labels)
    dataset = TwoObjectDataset(embeddings_tensor, labels_tensor)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    return dataloader, embeddings_tensor, labels_tensor

class TwoObjectClassifier(nn.Module):
    def __init__(self, input_dim, shared_dim):
        super(TwoObjectClassifier, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.adj_layers_1 = nn.Sequential(
            nn.Linear(shared_dim, 8),
            nn.Sigmoid()
        )
        
        self.noun_layers_1 = nn.Sequential(
            nn.Linear(shared_dim, 3),
            nn.Sigmoid()
        )
        
        self.adj_layers_2 = nn.Sequential(
            nn.Linear(shared_dim, 8),
            nn.Sigmoid()
        )
        
        self.noun_layers_2 = nn.Sequential(
            nn.Linear(shared_dim, 3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        shared = self.shared_layers(x)
        adj_out_1 = self.adj_layers_1(shared)
        noun_out_1 = self.noun_layers_1(shared)
        adj_out_2 = self.adj_layers_2(shared)
        noun_out_2 = self.noun_layers_2(shared)
        
        return adj_out_1, noun_out_1, adj_out_2, noun_out_2

def find_optimal_threshold(y_true, y_pred):
    thresholds = np.arange(0.1, 1, 0.05)
    best_threshold = 0.5
    best_score = 0
    for threshold in thresholds:
        score = f1_score(y_true, (y_pred > threshold).astype(int), average='micro')
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold

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
    learning_rate = 5e-05
    wt_decay = 1e-4
    num_epochs = 20
    
    model = TwoObjectClassifier(input_dim, shared_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wt_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    best_val_loss = float('inf')
    no_improve_epochs = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            inputs, (labels1, labels2) = batch
            adj_labels_1 = labels1[:, :8]
            noun_labels_1 = labels1[:, 8:11]
            adj_labels_2 = labels2[:, :8]
            noun_labels_2 = labels2[:, 8:11]

            adj_out_1, noun_out_1, adj_out_2, noun_out_2 = model(inputs)
            loss_adj_1 = criterion(adj_out_1, adj_labels_1)
            loss_noun_1 = criterion(noun_out_1, noun_labels_1)
            loss_adj_2 = criterion(adj_out_2, adj_labels_2)
            loss_noun_2 = criterion(noun_out_2, noun_labels_2)
            loss = loss_adj_1 + loss_noun_1 + loss_adj_2 + loss_noun_2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_dataloader):.4f}")

        model.eval()
        val_loss = 0
        for batch in val_dataloader:
            inputs, (labels1, labels2) = batch
            adj_labels_1 = labels1[:, :8]
            noun_labels_1 = labels1[:, 8:11]
            adj_labels_2 = labels2[:, :8]
            noun_labels_2 = labels2[:, 8:11]

            adj_out_1, noun_out_1, adj_out_2, noun_out_2 = model(inputs)
            loss_adj_1 = criterion(adj_out_1, adj_labels_1)
            loss_noun_1 = criterion(noun_out_1, noun_labels_1)
            loss_adj_2 = criterion(adj_out_2, adj_labels_2)
            loss_noun_2 = criterion(noun_out_2, noun_labels_2)
            loss = loss_adj_1 + loss_noun_1 + loss_adj_2 + loss_noun_2

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

        scheduler.step(val_loss)
    
    return model

def evaluate_model(model, data_loader):
    model.eval()
    all_predictions1 = []
    all_predictions2 = []
    all_true_labels1 = []
    all_true_labels2 = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, (labels1, labels2) = batch
            adj_out_1, noun_out_1, adj_out_2, noun_out_2 = model(inputs)

            all_predictions1.extend(torch.cat([adj_out_1, noun_out_1], dim=1).numpy())
            all_predictions2.extend(torch.cat([adj_out_2, noun_out_2], dim=1).numpy())
            all_true_labels1.extend(labels1.numpy())
            all_true_labels2.extend(labels2.numpy())

    threshold1 = find_optimal_threshold(np.array(all_true_labels1), np.array(all_predictions1))
    threshold2 = find_optimal_threshold(np.array(all_true_labels2), np.array(all_predictions2))

    predictions1 = (np.array(all_predictions1) > threshold1).astype(int)
    predictions2 = (np.array(all_predictions2) > threshold2).astype(int)
    all_trues = list(chain.from_iterable(zip(all_true_labels1, all_true_labels2)))
    all_preds = list(chain.from_iterable(zip(predictions1, predictions2)))

    return all_trues, all_preds

def calculate_metrics(true_labels, pred_labels):
    n_images = len(true_labels) // 2
    correct_label_predictions = 0
    total_labels = 0
    total_h_loss = 0
    total_f1 = 0
    
    for i in range(0, len(true_labels), 2):
        total_labels += 2
        
        # Check label-wise for each image for accuracy
        if np.array_equal(true_labels[i], pred_labels[i]) or np.array_equal(true_labels[i], pred_labels[i+1]):
            correct_label_predictions += 1
        if np.array_equal(true_labels[i+1], pred_labels[i]) or np.array_equal(true_labels[i+1], pred_labels[i+1]):
            correct_label_predictions += 1
        
        # Hamming loss and F1 score for the first true label
        hamming_distance_1 = np.sum(true_labels[i] != pred_labels[i])
        hamming_distance_2 = np.sum(true_labels[i] != pred_labels[i+1])
        if hamming_distance_1 < hamming_distance_2:
            h_loss_1 = hamming_loss(true_labels[i], pred_labels[i])
            f1_1 = f1_score(true_labels[i], pred_labels[i], average='micro')
        else:
            h_loss_1 = hamming_loss(true_labels[i], pred_labels[i+1])
            f1_1 = f1_score(true_labels[i], pred_labels[i+1], average='micro')

        # Hamming loss and F1 score for the second true label
        hamming_distance_1 = np.sum(true_labels[i+1] != pred_labels[i])
        hamming_distance_2 = np.sum(true_labels[i+1] != pred_labels[i+1])
        if hamming_distance_1 < hamming_distance_2:
            h_loss_2 = hamming_loss(true_labels[i+1], pred_labels[i])
            f1_2 = f1_score(true_labels[i+1], pred_labels[i], average='micro')
        else:
            h_loss_2 = hamming_loss(true_labels[i+1], pred_labels[i+1])
            f1_2 = f1_score(true_labels[i+1], pred_labels[i+1], average='micro')

        total_h_loss += (h_loss_1 + h_loss_2) / 2  # Averaging for the image
        total_f1 += (f1_1 + f1_2) / 2  # Averaging for the image
    
    average_h_loss = total_h_loss / n_images
    average_f1 = total_f1 / n_images
    accuracy = correct_label_predictions / total_labels    
    
    return {
        "accuracy": accuracy,
        "hamming_loss": average_h_loss,
        "f1_micro": average_f1
    }

def store_metrics(model_name, metrics_dict):
    with open(f"{model_name}_2obj_metrics.json", "w") as f:
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

with open("model_results/two_object_metrics.json", "w") as f:
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
    plt.savefig('model_results/two_object_cm.jpg', format='jpg', dpi=300, bbox_inches='tight')

mcm = compute_adapted_confusion_matrices(true_labs, pred_labs)

plot_multilabel_confusion_matrices_grid(mcm)

