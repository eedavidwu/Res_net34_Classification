# -*- coding:utf-8 -*-
import os
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import numpy as np


class Worker(data.Dataset):
    
    def __init__(self,root,transforms=None,train=True):
        '''
        Get images, divide into train/val set
        '''
        self.train = train
        self.images_root = root

        self._read_txt_file()
    
        if transforms is None:
            #normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
            #                        std=[0.5, 0.5, 0.5])

            if not train:
                self.transforms = T.Compose([
                    #T.CenterCrop((178, 178)),
                    T.Resize((96,96)),
                    #T.CenterCrop(224),
                    T.ToTensor(),
                    #normalize
                    ])

            else:
                self.transforms = T.Compose([
                    #T.CenterCrop((178, 178)),
                    T.Resize((96,96)),
                    #T.RandomResizedCrop(224),
                    #T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    #normalize
                    ])

    def get_mean_std(dataset):
        """Get mean and std by sample ratio
        """
        dataloader = data.DataLoader(dataset, batch_size=120,
                                     shuffle=True, num_workers=10)
        train = iter(dataloader).next()[0]
        mean = np.mean(train.numpy(), axis=(0, 2, 3))
        std = np.std(train.numpy(), axis=(0, 2, 3))
        return mean, std


    def _read_txt_file(self):
        self.images_path = []
        self.images_hat_labels = []
        self.images_cloth_labels = []

        if self.train:
            txt_file = os.path.join(self.images_root, 'Train.txt')
        else:
            txt_file = os.path.join(self.images_root, 'Val.txt')

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                self.images_path.append(item[0])
                self.images_hat_labels.append(item[1])
                self.images_cloth_labels.append(item[2])


        #print(txt_file)
    def __getitem__(self, index):
        '''
        return the data of one image
        '''
        img_path = self.images_path[index]
        label_hat = self.images_hat_labels[index]
        label_cloth=self.images_cloth_labels[index]
        data = Image.open(img_path)
        #print(data.size)
        data = self.transforms(data)
        #print('The data is:',data)
        #print('The label is:',label)
        return data, int(label_hat),int(label_cloth)
    
    def __len__(self):
        return len(self.images_path)
