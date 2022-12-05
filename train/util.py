
import math
import os
import random
from typing import Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import torch
import torch.autograd as autograd
from torch.autograd import Variable

np.set_printoptions(threshold=np.inf)

controls = [
    [0.331018, 0.637909, 0.150627, 0.561986, 0.671845, 0.946054, 0.309450, 0.081192, 0.335104, 0.644690, 0.073757, 0.102452, 0.321911, 0.156795],
    [0.069704, 0.515341, 0.625049, 0.729008, 0.761357, 0.608061, 0.232221, 0.486519, 0.689954, 0.796660, 0.376682, 0.734779, 0.106080, 0.052181],
    [0.551789, 0.558705, 0.418625, 0.473882, 0.777609, 0.587881, 0.535830, 0.660426, 0.838416, 0.764162, 0.561854, 0.383962, 0.435280, 0.839943]
]
image_names = [1, 1719, 2802]

conditions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device =  'cpu'

FloatTensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if device == 'cuda' else torch.LongTensor

# 生成对比图
'''
def sample_image(epochs, decoder, sample_image_path, image_path):
    """Saves a grid of generated digits ranging from 0 to n_classes"""

    control = Variable(FloatTensor(controls))
    gen_imgs = decoder(control)
    for i in range(gen_imgs.shape[0]):
        real_image = np.load(f'{image_path}{image_names[i]}.npy')
        fake_image = gen_imgs[i].to('cpu').data.numpy()
        comparison_image(real_image, fake_image, f'{sample_image_path}{epochs}_{i}.png')



def comparison_image(real_image: None, fake_image: None, path):
    cmap = plt.get_cmap('jet_r', 999999)
    cmap_error = plt.get_cmap('Reds', 999999)
    fig, axes = plt.subplots(nrows=3, ncols=5, sharex='col', sharey='col', figsize=(10, 6.6))
    error = np.abs(fake_image-real_image)
    real_image = np.ma.masked_outside(real_image, -0.1, 1.1)
    fake_image = np.ma.masked_outside(fake_image, -0.1, 1.1)

    sns.heatmap(real_image[0].data, vmin=0, vmax=1, ax=axes[0][0], cmap=cmap, cbar=False, mask=real_image[0].mask)
    sns.heatmap(real_image[1].data, vmin=0, vmax=1, ax=axes[1][0], cmap=cmap, cbar=False, mask=real_image[1].mask)
    sns.heatmap(real_image[2].data, vmin=0, vmax=1, ax=axes[2][0], cmap=cmap, cbar=False, mask=real_image[2].mask)

    sns.heatmap(fake_image[0].data, vmin=0, vmax=1, ax=axes[0][1], cmap=cmap, cbar=False, mask=fake_image[0].mask)
    sns.heatmap(fake_image[1].data, vmin=0, vmax=1, ax=axes[1][1], cmap=cmap, cbar=False, mask=fake_image[1].mask)
    sns.heatmap(fake_image[2].data, vmin=0, vmax=1, ax=axes[2][1], cmap=cmap, cbar=False, mask=fake_image[2].mask)

    sns.heatmap(error[0], vmin=-0, vmax=0.1, ax=axes[0][2], cmap=cmap_error, cbar=False)
    sns.heatmap(error[1], vmin=-0, vmax=0.1, ax=axes[1][2], cmap=cmap_error, cbar=False)
    sns.heatmap(error[2], vmin=-0, vmax=0.1, ax=axes[2][2], cmap=cmap_error, cbar=False)

    sns.kdeplot(real_image[0].flatten(), ax=axes[0][3], shade=True, label='Groud Truth')
    sns.kdeplot(fake_image[0].flatten(), ax=axes[0][3], shade=True, label='Generated')

    sns.kdeplot(real_image[0].flatten(), ax=axes[0][4], shade=True, cumulative=True, label='Groud Truth')
    sns.kdeplot(fake_image[0].flatten(), ax=axes[0][4], shade=True, cumulative=True, label='Generated')

    sns.kdeplot(real_image[1].flatten(), ax=axes[1][3], shade=True, label='Groud Truth')
    sns.kdeplot(fake_image[1].flatten(), ax=axes[1][3], shade=True, label='Generated')

    sns.kdeplot(real_image[1].flatten(), ax=axes[1][4], shade=True, cumulative=True, label='Groud Truth')
    sns.kdeplot(fake_image[1].flatten(), ax=axes[1][4], shade=True, cumulative=True, label='Generated')


    sns.kdeplot(real_image[2].flatten(), ax=axes[2][3], shade=True, label='Groud Truth')
    sns.kdeplot(fake_image[2].flatten(), ax=axes[2][3], shade=True, label='Generated')

    sns.kdeplot(real_image[2].flatten(), ax=axes[2][4], shade=True, cumulative=True, label='Groud Truth')
    sns.kdeplot(fake_image[2].flatten(), ax=axes[2][4], shade=True, cumulative=True, label='Generated')

    plt.subplots_adjust(left=0.02, bottom=0.05, right=0.98, top=0.95, wspace=0.15, hspace=0.25)

    axes[0][0].axis('off')
    axes[1][0].axis('off')
    axes[2][0].axis('off')
    axes[0][1].axis('off')
    axes[1][1].axis('off')
    axes[2][1].axis('off')
    axes[0][2].axis('off')
    axes[1][2].axis('off')
    axes[2][2].axis('off')

    axes[0][3].set_title('pressure-coefficient', fontsize=12)
    axes[1][3].set_title('x-velocity', fontsize=12)
    axes[2][3].set_title('y-velocity', fontsize=12)

    axes[0][3].yaxis.set_ticks([])
    axes[0][3].set_ylabel('Density', fontsize=10, rotation=90)
    axes[0][4].yaxis.set_ticks([])
    axes[0][4].set_ylabel('Cumulative')

    axes[1][3].yaxis.set_ticks([])
    axes[1][3].set_ylabel('Density', fontsize=10, rotation=90)
    axes[1][4].yaxis.set_ticks([])
    axes[1][4].set_ylabel('Cumulative')

    axes[2][3].yaxis.set_ticks([])
    axes[2][3].set_ylabel('Density', fontsize=10, rotation=90)
    axes[2][4].yaxis.set_ticks([])
    axes[2][4].set_ylabel('Cumulative')
    

    plt.savefig(path, dpi=600)
    plt.clf()
    plt.close()
'''

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha)
                    * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(FloatTensor(real_samples.shape[0], 1).fill_(
        1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates,inputs=interpolates,grad_outputs=fake,create_graph=True,retain_graph=True,only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# compute learning rate with decay in second half
def computeLR(i, epochs, minLR, maxLR):
    if i < epochs*0.5:
        return maxLR
    e = (i/float(epochs)-0.5)*2.
    # rescale second half to min/max range
    fmin = 0.
    fmax = 6.
    e = fmin + e*(fmax-fmin)
    f = math.pow(0.5, e)
    return minLR + (maxLR-minLR)*f


# add line to logfiles
def log(file, line, doPrint=True):
    f = open(file, "a+")
    f.write(line + "\n")
    f.close()
    if doPrint:
        print(line)


# reset log file
def resetLog(file):
    f = open(file, "w+")
    f.close()


def every_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = True

    
def statistics_param(decoder) -> Tuple[int, int] :
    Total_params = 0
    Trainable_params = 0
    # 遍历model.parameters() 返回的全局参数列表
    for param in decoder.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
    return Total_params, Trainable_params


from torch.utils.data import Dataset
import numpy as np
import glob
from enum import Enum

# 128*128
# max_min = (1.0537262,-2.49744,295.92245,-27.446018,192.1319,-119.62509)
# mean_std = (-0.139240873,0.360085847,98.22021867,27.17347675,2.899636259,14.70666339)
# 256*128
# max_min = (1.0537262,-2.49744,295.92245,-11.812475,192.1319,-119.62509)
# mean_std = (-0.221649127,0.410280127,104.0983921,25.7246567,5.795033549,16.61587769)

class Scope(Enum):
    all = 'all'
    self = 'self'
class Method(Enum):
    std = 0
    norm = 1
class PreProcess(Enum):
    method = Method
    scope = Scope

class DecoderDataset(Dataset):
    def __init__(self, image_path, label_path, tag, PreProcess:PreProcess):
        self.img_files = sorted(glob.glob('%s/*.npy' % image_path))
        self.label_files = sorted(glob.glob('%s/*.txt' % label_path))

    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)]
        img_name = os. img_path
        img = np.load(img_path)
        label = np.loadtxt(self.label_files[index % len(
            self.label_files)], dtype=np.float32, delimiter=',')

        img = self.PrePorcess(img,)
        return img, label

    def __normoalziation__(img, mp, dp, mx, dx, my, dy):
        img[0] = img[0] - mp / dp
        img[1] = img[1] - mx / dx
        img[2] = img[2] - my / dy
        return img

    def __len__(self):
        return len(self.img_files)



