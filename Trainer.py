from __future__ import print_function

import os
import numpy as np

import torch as t
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torchnet import meter
#from .TrainParams import TrainParams
from .log import logger
#from .visualize import Visualizer


def get_learning_rates(optimizer):
    lrs = [pg['lr'] for pg in optimizer.param_groups]
    lrs = np.asarray(lrs, dtype=np.float)
    return lrs

class Trainer(object):

    def __init__(self, model, params, train_data, val_data=None):
        # Data loaders
        self.train_data = train_data
        self.val_data = val_data
        # criterion and Optimizer and learning rate
        self.last_epoch = 0
        self.inital_epoch=0
        self.criterion = params.criterion
        self.optimizer = params.optimizer
        self.lr_scheduler = params.lr_scheduler
        logger.info('Set criterion to {}'.format(type(self.criterion)))
        logger.info('Set optimizer to {}'.format(type(self.optimizer)))
        logger.info('Set lr_scheduler to {}'.format(type(self.lr_scheduler)))

        # load model
        self.model = model
        logger.info('Set output dir to {}'.format(params.save_dir))
        if os.path.isdir(params.save_dir):
            pass
        else:
            os.makedirs(params.save_dir)

        ckpt = params.ckpt
        if ckpt is not None:
            self._load_ckpt(ckpt)
            logger.info('Load ckpt from {}'.format(ckpt))

        # meters
        self.loss_meter = meter.AverageValueMeter()
        self.confusion_matrix = meter.ConfusionMeter(9)

        # set CUDA_VISIBLE_DEVICES
        if len(params.gpus) > 0:
            gpus = ','.join([str(x) for x in params.gpus])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            params.gpus = tuple(range(len(params.gpus)))
            logger.info('Set CUDA_VISIBLE_DEVICES to {}...'.format(gpus))
            self.model = nn.DataParallel(self.model, device_ids=params.gpus)
            self.model = self.model.cuda()

        self.model.train()

    def train(self,params):
        #vis = Visualizer()
        best_loss = np.inf
        for epoch in range(self.inital_epoch, params.max_epoch):
            self.loss_meter.reset()
            self.confusion_matrix.reset()
            logger.info('Start training epoch {}'.format((self.last_epoch+1)))
            self._train_one_epoch(self.last_epoch, params)
            self.last_epoch += 1
    
            # save model
            if (self.last_epoch % params.save_freq_epoch == 0) or (self.last_epoch == params.max_epoch - 1):
                save_name = params.save_dir + 'ckpt_epoch_{}.pth'.format(self.last_epoch)
                t.save(self.model.state_dict(), save_name)

            val_cm, val_accuracy = self._val_one_epoch(params)

            if self.loss_meter.value()[0] < best_loss:
                logger.info('Found a better ckpt ({:.3f} -> {:.3f}), '.format(best_loss, self.loss_meter.value()[0]))
                best_loss = self.loss_meter.value()[0]
            '''
            # visualize
            vis.plot('loss', self.loss_meter.value()[0])
            vis.plot('val_accuracy', val_accuracy)
            vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
                epoch=epoch, loss=self.loss_meter.value()[0], val_cm=str(val_cm.value()),
                train_cm=str(self.confusion_matrix.value()), lr=get_learning_rates(self.optimizer)))
            '''
            # adjust the lr
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(self.loss_meter.value()[0], self.last_epoch)

    def _load_ckpt(self, ckpt):
        self.model.load_state_dict(t.load(ckpt))

    def _train_one_epoch(self, epoch, params):
        for step, (data, label) in enumerate(self.train_data):
            # train model
            inputs = Variable(data)
            target = Variable(label)
            #print(inputs.size())
            #print(target.size())
            if len(params.gpus) > 0:
                inputs = inputs.cuda()
                target = target.cuda()
            # forward
            score = self.model(inputs)
            loss = self.criterion(score, target)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            #print("***+++loss+++***:", loss)
            self.optimizer.step()
            # meters update
            self.loss_meter.add(loss.item())
            self.confusion_matrix.add(score.data, target.data)
            if not step % 50:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f \n'
                    % (epoch + 1, params.max_epoch, step,len(self.train_data), loss))
        #if not epoch % 5:
        #    print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
        #          % (epoch + 1, params.max_epoch, step,len(self.train_data), loss))
    def _val_one_epoch(self,params):
        self.model.eval()
        confusion_matrix = meter.ConfusionMeter(9)
        logger.info('Val on validation set...')

        for step, (data, label) in enumerate(self.val_data):

            # val model
            with t.no_grad():
                inputs = Variable(data)
                target = Variable(label.type(t.LongTensor))
            if len(params.gpus) > 0:
                inputs = inputs.cuda()
                target = target.cuda()

            score = self.model(inputs)
            confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))

        self.model.train()
        cm_value = confusion_matrix.value()

        accuracy = 100. * (cm_value[0][0] + cm_value[1][1]
                           + cm_value[2][2] + cm_value[3][3]
                           + cm_value[4][4] + cm_value[5][5]) / (cm_value.sum())
        return confusion_matrix, accuracy
