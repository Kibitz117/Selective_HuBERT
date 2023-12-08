import torch
import numpy as np
import os
from torch.utils.data import Dataset
import json
import csv
import librosa

def read_csv(csv_file_path):
    filenames = []
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                filenames.append(row[1])

    return filenames

def read_json_idx(json_file_path, vocab_list):
    """ 
    Returns a list of tuples consisting of a sample's filename and the 
    corresponding prediction index
    """
    samples = []
    
    with open(json_file_path, mode='r') as file:
        data = json.load(file)
        
    for sample in data.keys():
        vocab_idx = vocab_list.index(data[sample][0])
        samples.append((sample, vocab_idx))
    
    return samples

def flat_accuracy(preds, labels):
    pred_flat = torch.argmax(preds, dim=1)  # Returns indices of maximum values along the class dimension
    correct_predictions = pred_flat == labels  # Element-wise comparison
    accuracy = torch.sum(correct_predictions).item() / len(labels)  # Calculate accuracy
    return accuracy

class VocalImitationDataset(Dataset):
    """
    A dataset for the vocal imitation dataset from HEAR benchmarks.
    Assumes that the folder contains 3 subfolders delineating the folds, you'll make one for each fold
    Fold is a string representing the name of the fold json and folder
    This extracts the features for simplicity using the feature extractor passed
    """
    def __init__(self, data_dir, fold_name, vocab_file='labelvocabulary.csv', sample_rate='16000', extractor=None):
        vocab_path = os.path.join(data_dir, vocab_file)
        print(vocab_path)
        if os.path.exists(vocab_path):
            self.vocab_list = read_csv(vocab_path)
        else:
            raise Exception("Data folder must contain a valid vocab index csv file")
        
        fold_label_file = fold_name + '.json'
        label_path = os.path.join(data_dir, fold_label_file)
        print(label_path)
        if os.path.exists(label_path):
            self.label_index = read_json_idx(label_path, self.vocab_list)
        else:
            raise Exception("Data folder must contain a valid label index json file")
            
        self.fold_name = fold_name
        self.data_dir = data_dir
        self.sample_rate = sample_rate
            
    def __len__(self):
        return len(self.label_index)
    
    def __getitem__(self, idx):
        """
        Returns a numpy representing imitation wav file, and the index of the correct wav.
        """
        sample = self.label_index[idx]
        
        sample_path = os.path.join(self.data_dir, self.sample_rate, self.fold_name, sample[0])
        
        data, sr = librosa.load(sample_path, sr=None)
        assert sr == int(self.sample_rate)
        
        x = data
        y = sample[1]
        
        return [x, y]
    

class EmbeddingsDataset(Dataset):
    def __init__(self, data_dir, fold_name, vocab_file='labelvocabulary.csv'):
        vocab_path = os.path.join(data_dir, vocab_file)
        
        if os.path.exists(vocab_path):
            self.vocab_list = read_csv(vocab_path)
        else:
            raise Exception("Data folder must contain a valid vocab index csv file")

        fold_label_file = fold_name + '.json'
        label_path = os.path.join(data_dir, fold_label_file)
    
        with open(label_path, mode='r') as file:
            data = json.load(file)
            
        self.samples = list(data.keys())
        
        self.fold_name = fold_name
        self.data_dir = data_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx] + '.embedding.npy'
        embed_path = os.path.join(self.data_dir, self.fold_name, sample)
        embeddings = np.load(embed_path)
        
        embeddings = torch.from_numpy(embeddings)
        label = torch.tensor(int(sample[:3]), dtype=torch.int8)

        return [embeddings, label]


def train(dataloader, model, loss_fn, optimizer, device) -> float:
    size = len(dataloader.dataset)
    train_loss = 0.0

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        
        gen = model(X)
        loss = loss_fn(gen, y) 
        loss.backward()
        optimizer.step()

        # Append lists
        train_loss += loss.item()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return train_loss/len(dataloader)


def test(dataloader, model, loss_fn, device) -> float:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0.0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X,y = X.to(device), y.to(device)
            gen = model(X)
            test_loss += loss_fn(gen, y).item()

    test_loss /= num_batches

    print(
        f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    return test_loss

def selective_train(dataloader, model, selective_loss, optimizer, device) -> float:
    size = len(dataloader.dataset)
    train_loss = 0.0

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        logits, selection_logits, auxiliary_logits = model(X)

        #auxiliary_logits=auxiliary_logits.mean(dim=1)
        
        labels = y.long() # TODO just dix this in dataloader
        loss_dict = selective_loss(prediction_out=logits,
                                    selection_out=selection_logits,
                                    auxiliary_out=auxiliary_logits,
                                    target=labels)

        loss = loss_dict['loss']
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # Append lists
        train_loss += loss.item()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return train_loss/len(dataloader)


def selective_test(dataloader, model, device) -> float:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_acc = 0.0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits, selection_logits, auxiliary_logits = model(X)
            test_acc += flat_accuracy(logits, y)

    test_acc /= num_batches

    print(
        f"Test Error: \n Avg accuracy: {test_acc:>8f} \n")

    return test_acc
    