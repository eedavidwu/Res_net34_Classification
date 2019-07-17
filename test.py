#from __future__ import print_function

import os
from PIL import Image
from .log import logger

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F
from os import listdir, getcwd
wd = getcwd()

class Tester(object):
    def __init__(self, model, test_params):
        # load model
        self.model = model
        ckpt = test_params.ckpt
        if ckpt is not None:
            self._load_ckpt(ckpt)
            logger.info('Load ckpt from {}'.format(ckpt))

        # set CUDA_VISIBLE_DEVICES, 1 GPU is enough
        if len(test_params.gpus) > 0:
            gpu_test = str(test_params.gpus[0])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_test
            logger.info('Set CUDA_VISIBLE_DEVICES to {}...'.format(gpu_test))
            self.model = self.model.cuda()

        self.model.eval()
        print('Finish the initial')

    def test(self,params):
        img_list=open(params.testdata_dir)
        right_num=0
        all_num=0
        for line in img_list:
            line=line.split(' ')
            file_name=line[0].split('images')
            print('Processing image: ' , file_name[1])
            #image
            image_path=line[0]
            img = Image.open(image_path)
            #label
            true_label=line[1].strip()

            img = tv_F.to_tensor(tv_F.resize(img, (224, 224)))
            #img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img_input = Variable(torch.unsqueeze(img, 0))
            if len(params.gpus) > 0:
                img_input = img_input.cuda()

            output = self.model(img_input)
            score = F.softmax(output, dim=1)
            _, prediction = torch.max(score.data, dim=1)

            result_label_np = prediction.data.cpu().numpy()[0]
            print('Prediction label: ' , result_label_np)
            print('GroundTruth label: ' , true_label)
            all_num=all_num+1
            if (str(result_label_np) == true_label):
                print('---------Right-----------!!!')
                right_num=right_num+1
        map_rate=(right_num/all_num)
        print('mAP of the test images:%.3f%% (%d/%d)' % (map_rate*100,right_num,all_num))

    def _load_ckpt(self, ckpt):
        self.model=torch.nn.DataParallel(self.model)
        #cudnn.benchmark=True
        self.model.load_state_dict(torch.load(ckpt))
