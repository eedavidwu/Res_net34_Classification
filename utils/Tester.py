
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
        right_hat_cloth_num=0
        right_hat_num=0
        right_cloth_num=0
        right_no_hat_no_cloth_num=0
        all_num=0
        right_all_num=0
        for line in img_list:
            line=line.split(' ')
            file_name=line[0].split('images')
            #print('Processing image: ' , file_name[1])
            #image
            image_path=line[0]
            img = Image.open(image_path)
            #label
            true_hat_label=line[1]
            true_cloth_label = line[2].strip()
            img = tv_F.to_tensor(tv_F.resize(img, (96, 96)))
            #img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img_input = Variable(torch.unsqueeze(img, 0))
            if len(params.gpus) > 0:
                img_input = img_input.cuda()

            output = self.model(img_input)
            score_hat = F.softmax(output[0], dim=1)
            score_cloth = F.softmax(output[1], dim=1)
            _, prediction_hat = torch.max(score_hat.data, dim=1)
            _, prediction_cloth = torch.max(score_cloth.data, dim=1)

            result_label_hat_np = prediction_hat.data.cpu().numpy()[0]
            result_label_cloth_np = prediction_cloth.data.cpu().numpy()[0]
           
            all_num=all_num+1
            if (str(result_label_hat_np) == true_hat_label):
                right_hat_num = right_hat_num + 1
            if (str(result_label_cloth_np) == true_cloth_label):
                right_cloth_num = right_cloth_num + 1
            if ((str(result_label_hat_np) == true_hat_label) and (str(result_label_cloth_np) == true_cloth_label)):
                right_all_num = right_all_num + 1
            #else:
            #    print('Wrong!',result_label_hat_np,result_label_cloth_np,
            #         'Wanted:',true_hat_label,true_cloth_label)
            #   print(image_path)
        map_rate = (right_all_num / all_num)
        map_hat = right_hat_num / all_num
        map_cloth = right_cloth_num / all_num
        print('mAP of the all test images:%.3f%% (%d/%d)' % (map_rate * 100, right_all_num, all_num))
        print('mAP of the hat test:%.3f%% (%d/%d)' % (map_hat * 100, right_hat_num, all_num))
        print('mAP of the cloth: test %.3f%% (%d/%d)' % (map_cloth * 100, right_cloth_num, all_num))

    '''
            if (true_hat_label=='0' and true_cloth_label=='0'):
                if (str(result_label_hat_np) == true_hat_label and str(result_label_cloth_np) == true_cloth_label):
                    right_hat_cloth_num = right_hat_cloth_num + 1

            elif (true_hat_label=='0' and true_cloth_label=='1'):
                if (str(result_label_hat_np) == true_hat_label and str(result_label_cloth_np) == true_cloth_label):
                    right_hat_num = right_hat_num + 1

            elif (true_hat_label == '1' and true_cloth_label == '0'):
                if (str(result_label_hat_np) == true_hat_label and str(result_label_cloth_np) == true_cloth_label):
                    right_cloth_num = right_cloth_num + 1

            elif (true_hat_label == '1' and true_cloth_label == '1'):
                if (str(result_label_hat_np) == true_hat_label and str(result_label_cloth_np) == true_cloth_label):
                    right_no_hat_no_cloth_num = right_no_hat_no_cloth_num + 1

        right_all_num=right_no_hat_no_cloth_num+right_cloth_num+right_hat_num+right_hat_cloth_num
        map_rate=(right_all_num/all_num)
        map_hat=right_hat_num/300
        map_cloth=right_cloth_num/300
        map_hat_cloth= right_hat_cloth_num / 300
        map_no_hat_no_cloth = right_no_hat_no_cloth_num / 300
        print('mAP of the all test images:%.3f%% (%d/%d)' % (map_rate*100,right_all_num,all_num))
        print('mAP of the hat_no_cloth test:%.3f%% (%d/%d)' % (map_hat * 100, right_hat_num, 300))
        print('mAP of the cloth_no_hat: test %.3f%% (%d/%d)' % (map_cloth * 100, right_cloth_num, 300))
        print('mAP of the hat_cloth :%.3f%% (%d/%d)' % (map_hat_cloth * 100, right_hat_cloth_num, 300))
        print('mAP of the no_hat_no_cloth :%.3f%% (%d/%d)' % (map_no_hat_no_cloth * 100, right_no_hat_no_cloth_num, 300))
    '''

    def _load_ckpt(self, ckpt):
        self.model=torch.nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(ckpt))
