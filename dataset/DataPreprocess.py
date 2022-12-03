
import matplotlib.cm as cm
import os
from multiprocessing.pool import Pool

import matplotlib
import numpy as np
from PIL import Image
import imageio

os.chdir(os.path.dirname(__file__))

matplotlib.use('agg')

np.set_printoptions(threshold=np.inf)
from enum import Enum

class PreProcess(Enum):
    allnorm = 'allnorm'
    allstd = 'allstd'
    selfnorm = 'selfnorm'
    selfstd = 'selfstd'

# 控制点区间范围

# 取 region [top, bottom, left, right]
# 设定采样分辨率 (-region, region) 分成 x * y 个区间， 区间内点的均值做为区间的值 定义左上角 (0, 0) 
# region = [1.0, -1.0, -0.5, 1.5]
# x, y = 128, 128
# max_min = (1.0537262,-2.49744,295.92245,-27.446018,192.1319,-119.62509)
# mean_std = (-0.139240873,0.360085847,98.22021867,27.17347675,2.899636259,14.70666339)


region = [1.0, -1.0, -0.3, 0.7]
x, y = 256, 128
max_min = (1.0537262,-2.49744,295.92245,-11.812475,192.1319,-119.62509)
mean_std = (-0.221649127,0.410280127,104.0983921,25.7246567,5.795033549,16.61587769)

TYPE = 'CST'
method = PreProcess.allnorm
TAG = f'{x}_{y}_{method.value}'
CFDPath = f'{TYPE}/cfd_result/'
TrainPath = f'{TYPE}/{x}_{y}'
TargetPath = f'{TrainPath}Target/'
VisualPath = f'{TrainPath}Visual/'
FeaturePath = f'{TrainPath}Feature/'

os.makedirs(TargetPath) if not os.path.exists(TargetPath) else None
os.makedirs(VisualPath) if not os.path.exists(VisualPath) else None
os.makedirs(FeaturePath) if not os.path.exists(FeaturePath) else None

airfoils = np.loadtxt(f'{TYPE}Airfoil.csv', delimiter=',', skiprows=1)
params = np.loadtxt(f'{TYPE}Param.csv', delimiter=',', skiprows=1)

if method == PreProcess.allnorm:
    p_a, p_i, x_a, x_i, y_a, y_i = max_min
    d_p, d_vx, d_vy = p_a-p_i, x_a-x_i, y_a-y_i
elif method == PreProcess.allstd:
    p_i, d_p, x_i, d_vx, y_i, d_vy = mean_std


def data_extration(arg):
    if method == PreProcess.allnorm or method == PreProcess.allstd:
        global p_i, d_p, x_i, d_vx, y_i, d_vy
    Label = f'{arg[0]}_{arg[1]}'
    airfoil, param = airfoils[arg[1], 1:], params[arg[0], 1:]
    feature = np.append(airfoil, param, axis=None).reshape(1,-1)
    # 输出标签 
    np.savetxt(f'{FeaturePath}{Label}.txt', feature, delimiter=",", fmt='%0.9f')
    data = np.loadtxt(f'{CFDPath}{Label}.txt', dtype=np.float32, delimiter=",", skiprows=1)

    d_x = (region[3]-region[2]) / x
    d_y = (region[0]-region[1]) / y

    # 先选出区域 [top,bottom,left right]
    data = data[(region[2] <= data[:, 1]) & (data[:, 1] <= region[3]) & (region[1] <= data[:, 2]) & (data[:, 2] <= region[0]), :]
    boundary = data[np.where(data[:, 4] == 0)[0], 0]

    png = np.zeros((3, y, x))
    if method == PreProcess.selfnorm or method == PreProcess.allnorm:
        png = png - 1

    if method == PreProcess.selfnorm:
        p_a, p_i, x_a, x_i, y_a, y_i = np.max(data[:, 3]), np.min(data[:, 3]), np.max(data[:, 4]), np.min(data[:, 4]), np.max(data[:, 5]), np.min(data[:, 5])
        d_p, d_vx, d_vy = p_a-p_i, x_a-x_i, y_a-y_i
    elif method == PreProcess.selfstd:
        p_i, x_i, y_i = np.mean(data[:, 3]), np.mean(data[:, 4]), np.mean(data[:, 5])
        d_p, d_vx, d_vy = np.sqrt(np.std(data[:, 3])), np.sqrt(np.std(data[:, 4])), np.sqrt(np.std(data[:, 5]))

    i, j, t = 0, 0, 0
    b_p, b_x, b_y, b_n, b_t = 0, 0, 0, 0, 0
    while i < y and j < x:
        # 提取框内的data
        left, right, top, bottom = region[2] + (j-t)*d_x, region[2] + (j+t+1)*d_x, region[0] - (i-t)*d_y, region[0] - (i+t+1)*d_y
        temp = data[(left <= data[:, 1]) & (data[:, 1] <= right) & (bottom <= data[:, 2]) & (data[:, 2] <= top), :]
        n = len(temp)
        if n != 0:
            if t == 0:
                p, v_x, v_y = np.mean(temp[:, 3]), np.mean(temp[:, 4]), np.mean(temp[:, 5])
                png[0][i][j], png[1][i][j], png[2][i][j] = (p - p_i)/d_p, (v_x - x_i)/d_vx, (v_y - y_i)/d_vy
            elif t > 0:
                # 排除机翼内部点
                if np.intersect1d(temp[:, 0], boundary).size != 0:
                    if region[2] + (j+1)*d_x<=0 or region[2] + j*d_x>=1:
                        if b_t != 0 and b_p != 0:
                            p, v_x, v_y = b_p/b_n, b_x/b_n, b_y/b_n
                            png[0][i][j], png[1][i][j], png[2][i][j] = (p - p_i)/d_p, (v_x - x_i)/d_vx, (v_y - y_i)/d_vy
                            b_p, b_x, b_y, b_n, b_t, t = 0, 0, 0, 0, 0, 0
                        else:
                            p, v_x, v_y = np.mean(temp[:, 3]), np.mean(temp[:, 4]), np.mean(temp[:, 5])
                            png[0][i][j], png[1][i][j], png[2][i][j] = (p - p_i)/d_p, (v_x - x_i)/d_vx, (v_y - y_i)/d_vy
                    else:
                        t = 0
                elif b_p != 0:
                    if n > b_n:
                        p = b_p/b_n * t/(t+b_t) + (np.sum(temp[:, 3])-b_p)/(n-b_n)*b_t/(t+b_t)
                        v_x = b_x/b_n * t/(t+b_t)+ (np.sum(temp[:, 4])-b_x)/(n-b_n)*b_t/(t+b_t)
                        v_y = b_y/b_n * t/(t+b_t) + (np.sum(temp[:, 5])-b_y)/(n-b_n)*b_t/(t+b_t)
                        png[0][i][j], png[1][i][j], png[2][i][j] = (p - p_i)/d_p, (v_x - x_i)/d_vx, (v_y - y_i)/d_vy
                        b_p, b_x, b_y, b_n, b_t, t = 0, 0, 0, 0, 0, 0
                    else:
                        t += 1
                        continue
                else:
                    b_p, b_x, b_y, b_n, b_t, t = np.sum(temp[:, 3]), np.sum(temp[:, 4]), np.sum(temp[:, 5]), n, t, t+1
                    continue
            if j == x-1:
                i += 1
                j = 0
            else:
                j += 1
        else:
            t += 1

    # 输出通道数据
    np.save(file=f'{TargetPath}{Label}.npy', arr=png)

    # 输出通道数据的示意图
    cmap = cm.get_cmap('jet', 99999)
    if method == PreProcess.selfnorm or  method == PreProcess.allnorm:
        png = np.ma.masked_values(png, -1)
    elif method == PreProcess.selfstd or  method == PreProcess.allstd:
        png = np.ma.masked_values(png, 0)
    png[0] = matplotlib.colors.Normalize(vmin=np.min(png[0]), vmax=np.max(png[0]))(png[0])
    png[1] = matplotlib.colors.Normalize(vmin=np.min(png[1]), vmax=np.max(png[1]))(png[1])
    png[2] = matplotlib.colors.Normalize(vmin=np.min(png[2]), vmax=np.max(png[2]))(png[2])
    background = Image.new('RGBA', ((x*3 + (3-1)*1), y), (255, 255, 255, 0))
    pressure = Image.fromarray(cmap(png[0], bytes=True))
    background.paste(pressure, (0, 0))
    x_velocity = Image.fromarray(cmap(png[1], bytes=True))
    background.paste(x_velocity, (x + 1, 0))
    y_velocity = Image.fromarray(cmap(png[2], bytes=True))
    background.paste(y_velocity, (2*(x + 1), 0))
    background.save(fp=f'{VisualPath}{Label}.png')
    print(Label, '\tdone.')


def find_max():
    file = f'{TrainPath}statistics.csv'
    if not os.path.exists(file):
        f = open(file, 'w')
        print('','p_max', 'p_min', 'p_avg', 'p_std', 'p_var', 'p_mid', sep=",", end=',', file=f)
        print('vx_max', 'vx_min', 'vx_avg', 'vx_std', 'vx_var', 'vx_mid', sep=",", end=',', file=f)
        print('vy_max', 'vy_min', 'vy_avg', 'vy_std', 'vy_var', 'vy_mid',sep=",", file=f)
        for i in range(len(params)):
            for j in range(len(airfoils)):
                Label = f'{i}_{j}'
                data = np.loadtxt(f'{CFDPath}{Label}.txt', dtype=np.float32, delimiter=",", skiprows=1)
                # 先选出区域 [top,bottom,left right]
                data = data[(region[2] <= data[:, 1]) & (data[:, 1] <= region[3]) & (region[1] <= data[:, 2]) & (data[:, 2] <= region[0]), :]
                p_a = np.max(data[:, 3])
                p_i = np.min(data[:, 3])
                p_avg = np.mean(data[:, 3])
                p_std = np.std(data[:, 3])
                p_var = np.var(data[:, 3])
                p_mid = np.median(data[:, 3])

                x_a = np.max(data[:, 4])
                x_i = np.min(data[:, 4])
                x_avg = np.mean(data[:, 4])
                x_std = np.std(data[:, 4])
                x_var = np.var(data[:, 4])
                x_mid = np.median(data[:, 4])

                y_a = np.max(data[:, 5])
                y_i = np.min(data[:, 5])
                y_avg = np.mean(data[:, 5])
                y_std = np.std(data[:, 5])
                y_var = np.var(data[:, 5])
                y_mid = np.median(data[:, 5])

                print(Label, p_a, p_i, p_avg, p_std, p_var, p_mid, sep=",", end=',', file=f)
                print(x_a, x_i, x_avg, x_std, x_var, x_mid, sep=",", end=',', file=f)
                print(y_a, y_i, y_avg, y_std, y_var, y_mid, sep=",", file=f)
        f.close()




# find_max()
# find_control_scape()
# 测试样本
# data_extration((0, 0))


if __name__ == '__main__':
    # 多线程提取数据
    if 1: 
        args = []
        for i in range(len(params)):
            for j in range(len(airfoils)):
                if os.path.exists(f'{CFDPath}{i}_{j}.txt') and not os.path.exists(f'{TargetPath}{i}_{j}.npy'):
                    args.append((i,j))
        thread_pool = Pool()
        thread_pool.map_async(data_extration, args)
        thread_pool.close()
        thread_pool.join()
    # 输出流场图的视频
    if 1: 
        frames = []
        files  = os.listdir(VisualPath)
        for file in files:
            frames.append(imageio.imread(VisualPath+file))
        imageio.v2.mimsave(f'{TYPE}/{TAG}.mp4', frames)
    # 输出统计值
#     if 0: 
#         find_max()
