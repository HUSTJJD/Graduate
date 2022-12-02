
##
import os
import sys
BASE_DIR=  os.path.dirname(os.path.dirname( os.path.abspath(__file__) ))                   
# 将这个路径添加到环境变量中。
sys.path.append( BASE_DIR  )
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
####
import random
from datetime import datetime

import numpy as np
import torch

###
from model.TransGen import Decoder
from torch.autograd import Variable
from torch.utils.data import DataLoader
from train.util import FloatTensor, every_seed, log, resetLog, sample_image, weights_init_normal, statistics_param, DecoderDataset, device

from torchsummary import summary


# ----------
# path args
# ----------

TYPE = 'CST/128_128'
Task = 'transgen_self'

TargetPath = f'dataset/{TYPE}/Target'
FeaturePath = f'dataset/{TYPE}/Feature/'
TaskPath = f'result/{Task}/'
os.makedirs(TaskPath) if not os.path.isdir(TaskPath) else None

SaveModelPath = f'{TaskPath}Decoder'
RecordPath = f'{TaskPath}Record'
LogPath = f'{TaskPath}Monitor'

loss_record = 0.1

# ------------
# Random seed
# ------------
seed = random.randint(0, 2**32 - 1)
log(LogPath, f"[Random Seed: {seed}]")
every_seed(seed)

# --------------
# training args
# --------------

Epoch = 1000
BatchSize = 15
LearningRate = 0.0001



num_workers = 5
accumulated_step = 10 - BatchSize % 10 if BatchSize < 10 else 1
# Loss weight for gradient penalty
eval_ratio = 0.2

log(LogPath, f"[Epoch {Epoch}] [BatchSize {BatchSize}] [LearningRate: {LearningRate}] [eval_ratio {eval_ratio}]")


# ---------------------
# create train_dataset
# ---------------------
dataset = DecoderDataset(TargetPath, FeaturePath)
eval_size = int(eval_ratio * len(dataset))
train_size = len(dataset) - eval_size
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])


# ----------------------
# Configure data loader
# ----------------------
train_loader = DataLoader(dataset=train_dataset, batch_size=BatchSize, num_workers=num_workers, shuffle=True)

eval_loader = DataLoader(dataset=eval_dataset, batch_size=BatchSize, num_workers=num_workers, shuffle=True)


# ----------------------------------------
# Initialize Decoder 
# ----------------------------------------
decoder = Decoder().to(device)

# summary(Decoder, input_size=(1,12), batch_size=1)



# ------------
# Optimizers
# ------------
optimizer = torch.optim.Adam(decoder.parameters(), lr=LearningRate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode='min', patience=2, verbose=True)


# ----------------
# Loss functions
# ----------------
loss_func = torch.nn.L1Loss().to(device)



# --------------------
#  Load history model
# --------------------
try:
    decoder.load_state_dict(torch.load(f'{TaskPath}Decoder'))
    f = open(RecordPath, 'r+')
    loss_record = float(f.read())
    f.close()
except Exception:
    resetLog(LogPath) if not os.path.exists(SaveModelPath) else None
    Total_params, Trainable_params = statistics_param(decoder)
    log(LogPath, f'Total params: {Total_params}')
    log(LogPath, f'Trainable params: {Trainable_params}')

# ----------
#  Training
# ----------

StartTime = datetime.now()
log(LogPath, '[Start Time: {}]'.format(StartTime))


for epo in range(Epoch):
    loss_ep = 0
    decoder.train()
    for i, (target, feature) in enumerate(train_loader):
        target = Variable(target.type(FloatTensor))
        feature = Variable(feature.type(FloatTensor))
        # -----------------
        #  Train Decoder
        # -----------------
        optimizer.zero_grad()
        predict = decoder(feature)
        loss_bt = loss_func(target, predict)

        loss_bt.backward()
        loss_ep += loss_bt.sum()

        optimizer.step()

        if i % (len(train_loader)//2 + 1 or 2) == 0:
            print('[epo %d/%d] [Batch %d/%d] [Train loss: %f]' % (epo, Epoch, i, len(train_loader), loss_bt.item()))
    loss_ep /= len(train_loader.dataset)

    decoder.eval()
    loss_evep = 0
    with torch.no_grad():
        for i, (target, feature) in enumerate(eval_loader):
            target = Variable(target.type(FloatTensor))
            feature = Variable(feature.type(FloatTensor))

            predict = decoder(feature)
            loss_ev = loss_func(target, predict)

            loss_evep += loss_ev.sum()
            if i % (len(eval_loader) + 1 or 2) == 0:
                print('[epo %d/%d] [Batch %d/%d] [Eval loss: %f]' % (epo, Epoch, i, len(eval_loader), loss_ev.item()))

    loss_evep /= len(eval_loader.dataset)
    # test and save sample images
    # if epo % 10 == 0:
    #     sample_image(epochs=epo, decoder=decoder, sample_image_path=TaskPath, image_path=TargetPath)

    loss_mean = loss_mean
    scheduler.step(loss_mean)
    # average loss_ep
    log(LogPath, '[epo %d\t/%d] [Train loss: %f\t] [Eval loss: %f\t] [Avg loss: %f\t]' % (epo, Epoch, loss_ep, loss_evep, loss_mean), False)

    # save model and loss
    if loss_mean < loss_record:
        loss_record = loss_mean
        torch.save(obj=decoder.state_dict(), f=SaveModelPath)
        f = open(RecordPath, 'w')
        print(loss_record, file=f)
        f.close()

log(LogPath, '[Time Costed: {}]'.format(datetime.now() - StartTime))

