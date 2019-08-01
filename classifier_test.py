from torch import nn
from utils import Tester
from network import resnet34, resnet18, resnet50

import torch
import numpy as np
from utils import Tester
from utils import TestParams
import os
from os import listdir, getcwd
wd = getcwd()

# Set Test parameters
params = TestParams()
params.gpus = [0]  # set 'params.gpus=[]' to use CPU model. if len(params.gpus)>1, default to use params.gpus[0] to test
params.ckpt = './models/model_34/ckpt_epoch_40.pth'  #'./models/ckpt_epoch_400_res34.pth'
#params.testdata_dir = os.path.join(wd,'data','DIY_Test.txt')
params.testdata_dir = os.path.join(wd,'data','Test.txt')

##model:
model = resnet34(pretrained=False)  # batch_size=120, 1GPU Memory < 7000M
#model = resnet18(pretrained=False)  # batch_size=120, 1GPU Memory < 7000M
#model = resnet50(pretrained=False)  # batch_size=120, 1GPU Memory < 7000M
#print(model)

# Test
tester = Tester(model, params)
tester.test(params)

