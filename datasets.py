import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from PIL import  Image
import numpy as np
from scipy.io import loadmat
import os

DATAPATH = 'datasets/stanford_cars/'

def default_loader(path):
    try:
        img = Image.open(path).convert('RGB')
    except:
        with open('read_error.txt', 'a') as fid:
            fid.write(path+'\n')
        return Image.new('RGB', (224,224), 'white')
    return img

class TestDataset(Dataset):
    def __init__(self, dataloader = default_loader, transform = None):
        self.dataloader = dataloader
        self.transform = transform
        self.images = []
        self.labels = []
        data_list = open(DATAPATH + f'test_list.txt').readlines()
        for i, ss in enumerate(data_list):
            path, label = 'cars_train/' + ss.split(' ')[0], ss.split(' ')[1]
            self.images.append(path)
            self.labels.append(int(label))

    def __getitem__(self, index):
        image_path = os.path.join(DATAPATH, self.images[index])
        img = self.dataloader(image_path)
        img = self.transform(img)
        label = self.labels[index]
        return [img, label]

    def __len__(self):
        return len(self.images)



class RandomDataset(Dataset):
    def __init__(self, phase = 'train', transform=None, dataloader=default_loader):
        self.transform = transform
        self.dataloader = dataloader
        self.images = []
        self.labels = []
        data_list = open(DATAPATH + f'{phase}_list.txt').readlines()
        for i, ss in enumerate(data_list):
            path, label = 'cars_train/' + ss.split(' ')[0], ss.split(' ')[1]
            self.images.append(path)
            self.labels.append(int(label))

        self.labels = np.array(self.labels)
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, index):
        image_path = os.path.join(DATAPATH, self.images[index])
        img = self.dataloader(image_path)
        img = self.transform(img)
        label = self.labels[index]

        return [img, label]

    def __len__(self):
        return len(self.images)

class BatchDataset(Dataset):
    def __init__(self, phase, transform=None, dataloader=default_loader):
        self.transform = transform
        self.dataloader = dataloader
        self.images = []
        self.labels = []
        data_list = open(DATAPATH + f'{phase}_list.txt').readlines()
        for i, ss in enumerate(data_list):
            path, label = 'cars_train/' + ss.split(' ')[0], ss.split(' ')[1]
            self.images.append(path)
            self.labels.append(int(label))

        self.labels = np.array(self.labels)
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, index):
        image_path = os.path.join(DATAPATH, self.images[index])
        img = self.dataloader(image_path)
        img = self.transform(img)
        label = self.labels[index]

        return [img, label]


    def __len__(self):
        return len(self.images)

class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, n_classes, n_samples):
        self.labels = dataset.labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
