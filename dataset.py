# -*- coding:utf-8 -*-
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class Worker(data.Dataset):
    
    def __init__(self,root,transforms=None,train=True):
        '''
        Get images, divide into train/val set
        '''
        self.train = train
        self.images_root = root

        self._read_txt_file()
    
        if transforms is None:
            #normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
            #                        std=[0.229, 0.224, 0.225])

            if not train: 
                self.transforms = T.Compose([
                    #T.CenterCrop((178, 178)),
                    T.Resize((224,224)),
                    #T.CenterCrop(224),
                    T.ToTensor(),
                    #normalize
                    ]) 
            else:
                self.transforms = T.Compose([
                    #T.CenterCrop((178, 178)),
                    T.Resize((224,224)),
                    #T.RandomResizedCrop(224),
                    #T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    #normalize
                    ])
                
    def _read_txt_file(self):
        self.images_path = []
        self.images_labels = []

        if self.train:
            txt_file = os.path.join(self.images_root, 'Train.txt')
        else:
            txt_file = os.path.join(self.images_root, 'Val.txt')

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                self.images_path.append(item[0])
                self.images_labels.append(item[1])

        #print(txt_file)
    def __getitem__(self, index):
        '''
        return the data of one image
        '''
        img_path = self.images_path[index]
        label = self.images_labels[index]
        data = Image.open(img_path)
        #print(data.size)
        data = self.transforms(data)
        #print('The data is:',data)
        #print('The label is:',label)
        return data, int(label)
    
    def __len__(self):
        return len(self.images_path)
