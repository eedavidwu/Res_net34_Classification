class TestParams(object):
    def __init__(self):
        self.gpus = None  # default to use CPU mode
        self.ckpt = None                 # path to the pretrained ckpt file
        self.testdata_dir = None  # default `save_dir`

params = TestParams()
