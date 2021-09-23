from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "arcface"
config.network = "r100" # I changed
config.resume = True # I changed
config.output = "output" # I changed
config.embedding_size = 512

config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.0001  # I changed

config.rec = "/home/hankyul/hdd_ext/face/synthetic"
config.num_classes = 93979
config.num_image = 2459393
config.num_epoch = 25
config.warmup_epoch = -1
config.decay_epoch = [10, 16, 22]
config.val_targets = ["lfw", "cfp_fp"]