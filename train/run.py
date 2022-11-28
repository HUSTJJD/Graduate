
##
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
####
import random
from datetime import datetime

import numpy as np
import torch

###
from model.TransGen import Decoder
from torch.autograd import Variable
from torch.utils.data import DataLoader
from train.util import FloatTensor, every_seed, log, resetLog, sample_image, weights_init_normal, statistics_param, DecoderDataset



from torchsummary import summary


# ----------
# path args
# ----------

TYPE = 'CST/256_128'
Task = 'transgen_self'

TargetPath = f'dataset/{TYPE}/Target'
FeaturePath = f'dataset/{TYPE}/Feature/'
TaskPath = f'result/{Task}/'
os.mkdir(TaskPath) if not os.path.isdir(TaskPath) else None

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
BatchSize = 10
LearningRate = 0.0001



num_workers = 10
accumulated_step = 10 - BatchSize % 10 if BatchSize < 10 else 1
# Loss weight for gradient penalty
eval_ratio = 0.2

log(LogPath, f"[Epoch {Epoch}] [BatchSize {BatchSize}] [LearningRate: {LearningRate}] [eval_ratio {eval_ratio}]")

# ----------------------
# Configure device auto
# ----------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
train_loader = DataLoader(dataset=train_dataset, batch_size=BatchSize, num_workers=2, shuffle=True)

eval_loader = DataLoader(dataset=eval_dataset, batch_size=BatchSize, num_workers=2, shuffle=True)


# ----------------------------------------
# Initialize Decoder 
# ----------------------------------------
decoder = Decoder().to(device)

summary(Decoder, input_size=(1,14), BatchSize=1)
Total_params, Trainable_params = statistics_param(decoder)
log(LogPath, f'Total params: {Total_params}')
log(LogPath, f'Trainable params: {Trainable_params}')


# ------------
# Optimizers
# ------------
optimizer = torch.optim.Adam(decoder.parameters(), LearningRate=LearningRate)
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


# ----------
#  Training
# ----------
StartTime = datetime.now()
log(LogPath, '[Start Time: {}]'.format(StartTime))
decoder.train()
for epo in range(Epoch):
    loss_ep = 0

    for i, (imgs, labels) in enumerate(train_loader):
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(FloatTensor))
        # -----------------
        #  Train Decoder
        # -----------------
        optimizer.zero_grad()
        fake_imgs = decoder(labels)
        loss_bt = loss_func(real_imgs, fake_imgs)

        loss_bt.backward()
        loss_ep += loss_bt.item()*imgs.shape[0]

        optimizer.step()

        if i % (len(train_loader)//2 + 1 or 2) == 0:
            print('[epo %d/%d] [Batch %d/%d] [Train loss: %f]' % (epo, Epoch, i, len(train_loader), loss_bt.item()))

    loss_ep /= len(train_loader.dataset)

    decoder.eval()
    loss_evep = 0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(eval_loader):
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(FloatTensor))

            fake_imgs = decoder(labels)
            loss_ev = loss_func(real_imgs, fake_imgs)

            loss_evep += loss_ev.item()*imgs.shape[0]
            if i % (len(eval_loader) + 1 or 2) == 0:
                print('[epo %d/%d] [Batch %d/%d] [Eval loss: %f]' % (epo, Epoch, i, len(eval_loader), loss_ev.item()))

    loss_evep /= len(eval_loader.dataset)
    # test and save sample images
    if epo % 10 == 0:
        sample_image(Epochs=epo, decoder=decoder, sample_image_path=TaskPath, image_path=TargetPath)

    scheduler.step((loss_evep+loss_ep)/2)
    # average loss_ep
    log(LogPath, '[epo %d\t/%d] [Train loss: %f\t] [Eval loss: %f\t] [Avg loss: %f\t]' % (epo, Epoch, loss_ep, loss_evep, (loss_evep+loss_ep)/2), False)

    # save model and loss
    if (loss_evep+loss_ep)/2 < loss_record:
        loss_record = (loss_evep+loss_ep)/2
        torch.save(obj=decoder.state_dict(), f=SaveModelPath)
        f = open(RecordPath, 'w')
        print(loss_record, file=f)
        f.close()

    his_loss_ep = loss_ep
    his_loss_ev = loss_evep


log(LogPath, '[Time Costed: {}]'.format(datetime.now() - StartTime))


class Train():
    def __init__(self) -> None:
        pass

    def start(self) -> None:
        pass

    def end(self) -> None:
        pass