from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.cls_task = False
config.loss = "arcface"
config.network = "vit_base_r50_s16_224_in21k" # I changed
config.resume = False # I changed
config.output = "output" # I changed
config.embedding_size = 768

config.sample_rate = 1.0
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 64
config.lr = 0.1 # I changed

config.rec = "/home/hankyul/shared/face/unmask"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 25
config.warmup_epoch = 1
config.decay_epoch = [10, 16, 22]
config.val_targets = ["cfp_fp"]