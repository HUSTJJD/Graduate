import matplotlib.animation as animation
import math
import matplotlib.pyplot as plt
import numpy as np
from math import pi, cos, sin, factorial

import pandas as pd
import random
import os

'''
    NACA 4 位数翼型生成器
'''


def naca4(arg):
    dirpath, filename, m, p, t = arg[0], arg[1], arg[2], arg[3], arg[4]
    NACA4(dirpath, filename, m, p, t)


class NACA4():
    def __init__(self, dirpath, filename, m, p, t, N=400, c=1):
        self.dirpath, self.filename = dirpath, filename
        self.xu = []
        self.yu = []
        self.xl = []
        self.yl = []
        for x in np.linspace(0, c, N//2):
            dyc_dx = NACA4.dyc_over_dx(x, m, p, c)
            th = math.atan(dyc_dx)
            yt = NACA4.thickness(x, t, c)
            yc = NACA4.camber_line(x, m, p, c)
            self.xu.append(x - yt * math.sin(th) - c/2)
            self.yu.append(yc + yt * math.cos(th))
            self.xl.append(x + yt * math.sin(th) - c/2)
            self.yl.append(yc - yt * math.cos(th))
        self.xl = self.xl[::-1]
        self.yl = self.yl[::-1]
        self.__writeToFile()
        # self.__plotting()

    def __writeToFile(self):
        airfoil_shape_file = self.dirpath + os.path.sep + \
            'dat' + os.path.sep + self.filename + '.dat'
        coord_file = open(airfoil_shape_file, 'w')
        print(len(self.xl), file=coord_file)
        for i in range(len(self.xl)):
            print('{:<.12f} {:<.12f}'.format(
                float(self.xl[i]), float(self.yl[i])), float(0), file=coord_file)
        print(len(self.xu), file=coord_file)
        for i in range(len(self.xu)):
            print('{:<.12f} {:<.12f}'.format(
                float(self.xu[i]), float(self.yu[i])), float(0), file=coord_file)
        print(2, file=coord_file)
        print('{:<.12f} {:<.12f}'.format(
            float(self.xu[-1]), float(self.yu[-1])), float(0), file=coord_file)
        print('{:<.12f} {:<.12f}'.format(
            float(self.xl[0]), float(self.yl[0])), float(0), file=coord_file)
        coord_file.close()

    def __plotting(self):
        fig7 = plt.figure()
        ax7 = plt.subplot(111)
        ax7.plot(self.xu, self.yu)
        ax7.plot(self.xl, self.yl)
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.ylim(ymin=-0.75, ymax=0.75)
        ax7.spines['right'].set_visible(False)
        ax7.spines['top'].set_visible(False)
        ax7.yaxis.set_ticks_position('left')
        ax7.xaxis.set_ticks_position('bottom')
        airfoil_shape = self.dirpath + os.path.sep + \
            'dat' + os.path.sep + self.filename + '.png'
        plt.savefig(airfoil_shape)

    # https://en.wikipedia.org/wiki/NACA_airfoil#Equation_for_a_cambered_4-digit_NACA_airfoil
    @staticmethod
    def camber_line(x, m, p, c):
        if 0 <= x <= c * p:
            yc = m * (x / math.pow(p, 2)) * (2 * p - (x / c))
        # elif p * c <= x <= c:
        else:
            yc = m * ((c - x) / math.pow(1-p, 2)) * (1 + (x / c) - 2 * p)
        return yc

    @staticmethod
    def dyc_over_dx(x, m, p, c):
        if 0 <= x <= c * p:
            dyc_dx = ((2 * m) / math.pow(p, 2)) * (p - x / c)
        # elif p * c <= x <= c:
        else:
            dyc_dx = ((2 * m) / math.pow(1-p, 2)) * (p - x / c)
        return dyc_dx

    @staticmethod
    def thickness(x, t, c):
        term1 = 0.2969 * (math.sqrt(x/c))
        term2 = -0.1260 * (x/c)
        term3 = -0.3516 * math.pow(x/c, 2)
        term4 = 0.2843 * math.pow(x/c, 3)
        term5 = -0.1015 * math.pow(x/c, 4)
        return 5 * t * c * (term1 + term2 + term3 + term4 + term5)


'''
    CST 翼型生成器
'''


def cst(arg):
    dirpath, filename, wu, wl = arg[0], arg[1], arg[2:8], arg[8:]
    CST(dirpath, filename, wl, wu)


class CST():
    def __init__(self, dirpath, filename, wl, wu, dz=0.001, N=400):
        # wl = [-1, -1, -1], wu = [1, 1, 1]
        self.dirpath, self.filename, self.wl, self.wu = dirpath, filename, wl, wu
        self.dz = dz
        self.N = N
        self.airfoil_coor()

    def airfoil_coor(self):
        # Create x coordinate
        x = np.ones((self.N, 1))
        zeta = np.zeros((self.N, 1))

        for i in range(0, self.N):
            zeta[i] = 2 * pi / self.N * i
            x[i] = 0.5*(cos(zeta[i])+1)

        # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
        N1 = 0.5
        N2 = 1
        # Used to separate upper and lower surfaces
        center_loc = np.where(x == 0)
        center_loc = center_loc[0][0]

        self.xl = np.zeros(center_loc)
        self.xu = np.zeros(self.N-center_loc)

        for i in range(len(self.xl)):
            self.xl[i] = x[i]            # Lower surface x-coordinates
        for i in range(len(self.xu)):
            self.xu[i] = x[i + center_loc]   # Upper surface x-coordinates
        self.xl[-1] = 0.0
        assert len(self.xl) == len(self.xu)

        # Call ClassShape function to determine lower surface y-coordinates
        self.yl = self.__ClassShape(self.wl, self.xl, N1, N2, -self.dz)
        # Call ClassShape function to determine upper surface y-coordinates
        self.yu = self.__ClassShape(self.wu, self.xu, N1, N2, self.dz)

        # self.__plotting()
        self.__writeToFile()

    def inv_airfoil_coor(self, x):
        # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
        N1 = 0.5
        N2 = 1

        # Used to separate upper and lower surfaces
        center_loc = np.where(x == 0)
        center_loc = center_loc[0][0]

        xl = np.zeros(center_loc)
        xu = np.zeros(self.N-center_loc)

        for i in range(len(xl)):
            xl[i] = x[i]            # Lower surface x-coordinates
        for i in range(len(xu)):
            xu[i] = x[i + center_loc]   # Upper surface x-coordinates

        # Call ClassShape function to determine lower surface y-coordinates
        self.yl = self.__ClassShape(self.wl, self.xl, N1, N2, -self.dz)
        # Call ClassShape function to determine upper surface y-coordinates
        self.yu = self.__ClassShape(self.wu, self.xu, N1, N2, self.dz)

        self.__writeToFile()

    # Function to calculate class and shape function
    @staticmethod
    def __ClassShape(w, x, N1, N2, dz):
        # Class function; taking input of N1 and N2
        C = np.zeros(len(x))
        for i in range(len(x)):
            C[i] = x[i]**N1*((1-x[i])**N2)

        # Shape function; using Bernstein Polynomials
        n = len(w) - 1  # Order of Bernstein polynomials

        K = np.zeros(n+1)
        for i in range(0, n+1):
            K[i] = factorial(n)/(factorial(i)*(factorial((n)-(i))))

        S = np.zeros(len(x))
        for i in range(len(x)):
            S[i] = 0
            for j in range(0, n+1):
                S[i] += w[j]*K[j]*x[i]**(j) * ((1-x[i])**(n-(j)))

        # Calculate y output
        y = np.zeros(len(x))
        for i in range(len(y)):
            y[i] = C[i] * S[i] + x[i] * dz

        return y

    def __writeToFile(self):
        airfoil_shape_file = self.dirpath + os.path.sep + \
            'dat' + os.path.sep + self.filename + '.dat'
        coord_file = open(airfoil_shape_file, 'w')
        print(len(self.xl), file=coord_file)
        for i in range(len(self.xl)):
            print('{:<.12f} {:<.12f}'.format(
                float(self.xl[i]), float(self.yl[i])), float(0), file=coord_file)
        print(len(self.xu), file=coord_file)
        for i in range(len(self.xu)):
            print('{:<.12f} {:<.12f}'.format(
                float(self.xu[i]), float(self.yu[i])), float(0), file=coord_file)
        print(2, file=coord_file)
        print('{:<.12f} {:<.12f}'.format(
            float(self.xu[-1]), float(self.yu[-1])), float(0), file=coord_file)
        print('{:<.12f} {:<.12f}'.format(
            float(self.xl[0]), float(self.yl[0])), float(0), file=coord_file)
        coord_file.close()

    def __plotting(self):
        fig7 = plt.figure()
        ax7 = plt.subplot(111)
        ax7.plot(self.xu, self.yu)
        ax7.plot(self.xl, self.yl)
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.ylim(ymin=-0.75, ymax=0.75)
        ax7.spines['right'].set_visible(False)
        ax7.spines['top'].set_visible(False)
        ax7.yaxis.set_ticks_position('left')
        ax7.xaxis.set_ticks_position('bottom')
        airfoil_shape = self.dirpath + os.path.sep + \
            'dat' + os.path.sep + self.filename + '.png'
        plt.savefig(airfoil_shape)


'''
    拉丁超立方采样：
    1.接收到一组变量范围numpy矩阵以及样本需求个数，shape = (m,2)，输出样本numpy矩阵
    执行ParameterArray函数即可
'''


class DoE(object):
    def __init__(self, name_value, bounds, seed):
        self.name = name_value
        self.bounds = bounds
        self.type = "DoE"
        self.result = None
        random.seed = seed
        np.random.seed = seed


class DoE_LHS(DoE):
    # 拉丁超立方试验样本生成
    def __init__(self, name_value, bounds, N, seed=990526):
        DoE.__init__(self, name_value, bounds, seed)
        self.type = "LHS"
        self.parameterArray = DoE_LHS.ParameterArray(bounds, N)
        self.N = N

    def write_to_csv(self, filename):
        # 将样本数据写入LHS.csv文件，文件保存至运行文件夹内
        sample_data = pd.DataFrame(self.parameterArray, columns=self.name)
        sample_data.to_csv(filename)

    @staticmethod
    def ParameterArray(limitArray, sampleNumber):
        # 根据输入的各变量的范围矩阵以及希望得到的样本数量，输出样本参数矩阵
        # :param limitArray:变量上下限矩阵，shape为(m,2),m为变量个数
        # :param sampleNumber:希望输出的 样本数量
        # :return:样本参数矩阵
        arr = DoE_LHS.Partition(sampleNumber, limitArray)
        parametersMatrix = DoE_LHS.Rearrange(DoE_LHS.Representative(arr))
        return parametersMatrix

    @staticmethod
    def Partition(number_of_sample, limit_array):
        # 为各变量的变量区间按样本数量进行划分，返回划分后的各变量区间矩阵
        # :param number_of_sample: 需要输出的 样本数量
        # :param limit_array: 所有变量范围组成的矩阵,为(m, 2)矩阵，m为变量个数，2代表上限和下限
        # :return: 返回划分后的个变量区间矩阵（三维矩阵），三维矩阵每层对应于1个变量
        coefficient_lower = np.zeros((number_of_sample, 2))
        coefficient_upper = np.zeros((number_of_sample, 2))
        for i in range(number_of_sample):
            coefficient_lower[i, 0] = 1 - i / number_of_sample
            coefficient_lower[i, 1] = i / number_of_sample
        for i in range(number_of_sample):
            coefficient_upper[i, 0] = 1-(i+1) / number_of_sample
            coefficient_upper[i, 1] = (i+1) / number_of_sample

        partition_lower = coefficient_lower @ limit_array.T  # 变量区间下限
        partition_upper = coefficient_upper @ limit_array.T  # 变量区间上限

        # 得到各变量的区间划分，三维矩阵每层对应于1个变量
        partition_range = np.dstack((partition_lower.T, partition_upper.T))
        return partition_range  # 返回区间划分上下限

    @staticmethod
    def Representative(partition_range):
        # 计算单个随机代表数的函数
        # :param partition_range: 一个shape为 (m,N,2) 的三维矩阵，m为变量个数、n为样本个数、2代表区间上下限的两列
        # :return: 返回由各变量分区后区间随机代表数组成的矩阵，每列代表一个变量
        number_of_value = partition_range.shape[0]  # 获得变量个数
        numbers_of_row = partition_range.shape[1]  # 获得区间/分层个数
        coefficient_random = np.zeros(
            (number_of_value, numbers_of_row, 2))  # 创建随机系数矩阵
        representative_random = np.zeros((numbers_of_row, number_of_value))

        for m in range(number_of_value):
            for i in range(numbers_of_row):
                y = random.random()
                coefficient_random[m, i, 0] = 1 - y
                coefficient_random[m, i, 1] = y

        # 利用*乘实现公式计算（对应位置进行乘积计算），计算结果保存于临时矩阵 temp_arr 中
        temp_arr = partition_range * coefficient_random
        for j in range(number_of_value):  # 计算每个变量各区间内的随机代表数，行数为样本个数n，列数为变量个数m
            temp_random = temp_arr[j, :, 0] + temp_arr[j, :, 1]
            representative_random[:, j] = temp_random
        return representative_random  # 返回代表数向量

    @staticmethod
    def Rearrange(arr_random):
        # 打乱矩阵各列内的数据
        # :param arr_random: 一个N行, m列的矩阵
        # :return: 每列打乱后的矩阵
        for i in range(arr_random.shape[1]):
            np.random.shuffle(arr_random[:, i])
        return arr_random


def PlotAirfoils(type):
    font = {'size': 12}
    fig = plt.figure(figsize=[5, 2.2])
    plt.axis('equal')
    plt.xlabel('x/c')
    plt.ylabel('y/c')
    plt.tight_layout()
    path = f'{type}/dat/'
    artists = []
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            f = open(root + file, 'r')
            lines = f.readlines()
            x, y = [], []
            for line in lines:
                arr = line.split(' ')
                if(len(arr) > 1):
                    x.append(float(arr[0]))
                    y.append(float(arr[1]))
            im = plt.plot(x, y, linewidth=1)
            artists.append(im)
    ani = animation.ArtistAnimation(
        fig=fig, artists=artists, interval=500, repeat=False)
    airfoil_shape = f'{type}' + os.path.sep + f'{type}' + '.mp4'
    ani.save(airfoil_shape, dpi=600)
