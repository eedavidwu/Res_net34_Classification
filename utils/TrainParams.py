class TrainParams(object):
    def __init__(self):
        self.max_epoch = 30
        self.criterion = None
        self.gpus = [0]  # default to use CPU mode
        self.save_dir = './models/'  # default `save_dir`
        self.ckpt = None                 # path to the pretrained ckpt file

        # saving checkpoints
        self.save_freq_epoch = 100  # save one ckpt per `save_freq_epoch` epochs

        # optimizer and criterion and learning rate scheduler
        self.optimizer = None
        self.lr_scheduler = None         # should be an instance of ReduceLROnPlateau or _LRScheduler

params = TrainParams()
