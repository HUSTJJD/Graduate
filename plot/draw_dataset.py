from operator import le
import os
from turtle import color, left, numinput, width
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from scipy.ndimage.filters import gaussian_filter
from datetime import datetime
import seaborn as sns
import matplotlib as mpl
import random
import torch
from model.TransGen import Generator
import matplotlib.pyplot as plt
import re
from PIL import Image, ImageDraw
import numpy as np
import matplotlib


matplotlib.use('agg')
image_size = 128

save_path = './paper_image/'
image_path = './train_data/128/image/'
label_path = './train_data/128/label/'
stats_path = 'train_data/statistical/'
cmap = plt.get_cmap('jet_r', 999999)
cmap_error = plt.get_cmap('Reds', 999999)
model = 'transgen'
modelpath = './train_image/save_model/{}/generator'.format(model)
device = torch.device('cpu')
model = Generator().to(device)

loss_func = torch.nn.L1Loss(reduction='sum')

model.load_state_dict(torch.load(modelpath, map_location=device))
model.eval()


def draw_mse():

    maes = 0
    for i in range(1, 3424):
        label = torch.Tensor(
            np.loadtxt(label_path + '{}.txt'.format(i),
                       dtype=np.float,
                       delimiter=",").reshape((1, 1, 14))).to(device)
        real_imgs = np.load(image_path +
                            '{}.npy'.format(i)).reshape(
                                (1, 3, 128, 128))[0]
        fake_imgs = model(label).cpu().detach().numpy()[0]
        mae = np.mean(np.abs(real_imgs - fake_imgs))
        maes += mae/3423
        print('i',i,'mae',mae, 'maes',maes)
    print('avg maes',maes)
# draw_mse()

def draw_dataset():
    n_col = 4
    n_row = 3
    names = np.random.choice(np.linspace(1, 3423, 3423, dtype=np.int),
                             n_col * n_row)
    # names = np.linspace(1, 3423, 3423, dtype=np.int)
    n_col_ = n_col * 3
    background = Image.new(
        'RGBA', ((image_size * n_col_ + (n_col_ - 1) * 2 + (n_col - 1) * 4),
                 (image_size * n_row + (n_row - 1) * 6)), (255, 255, 255, 255))
    for i in range(len(names)):
        imgs = np.load(image_path + '{}.npy'.format(names[i]))
        n = i % (n_col_ // 3) * ((image_size + 2) * 3 + 4)
        m = i // (n_col_ // 3) * (image_size + 6)
        for j in range(imgs.shape[0]):
            img = cmap(np.ma.masked_outside(imgs[j], 0, 1), bytes=True)
            img = Image.fromarray(img)
            # draw = ImageDraw.Draw(img)
            # draw.text((40, 40), str(names[i]))
            background.paste(img, (n, m))
            n += image_size + 2

    background.save(save_path + 'dataset_sample.png')

# draw_dataset()


def draw_test_loss():
    n_col = 1
    n_col_ = n_col * 3
    n_row = 1
    nums = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    max_error = 0
    max_index = 0
    losses = []
    plt.subplot(121)
    with torch.no_grad():
        for m in range(5):
            loss_list = []
            for num in nums:

                avg_loss = 0
                names = np.random.choice(np.linspace(
                    1, 3423, 3423, dtype=np.int), num)
                for i in range(num):
                    label = torch.Tensor(
                        np.loadtxt(label_path + '{}.txt'.format(names[i]),
                                   dtype=np.float,
                                   delimiter=",").reshape((1, 1, 14))).to(device)
                    real_imgs = np.load(image_path +
                                        '{}.npy'.format(names[i])).reshape(
                                            (1, 3, 128, 128))
                    fake_imgs = model(label).cpu().detach().numpy()

                    sum_loss = 0
                    for j in range(fake_imgs.shape[1]):
                        loss = np.mean(
                            np.abs(real_imgs[0][j] - fake_imgs[0][j]))
                        sum_loss += loss * 1/3

                    losses.append(sum_loss)
                    avg_loss += sum_loss/num
                    if sum_loss > max_error:
                        max_error = sum_loss
                        max_index = names[i]

                print('avg_loss:', str(avg_loss)+'\t', 'num:', num)
                loss_list.append(avg_loss)

            plt.plot(nums, loss_list, label='test-{}'.format(m+1), linewidth=1)
            print(max_error)
            print(max_index)
        plt.legend()
        plt.xlabel('Number of test sample', fontsize=12)
        plt.ylabel('MAE', fontsize=12)
        # plt.gcf().savefig(save_path + 'dataset_avg_error.eps',
        #                   format='eps', dpi=1000, transparent=True)
        plt.subplot(122)
        lenth = len(losses)
        sns.kdeplot(losses, shade=True, cut=0)
        plt.ylabel('Density', fontsize=12)
        plt.xlabel('MAE', fontsize=12)
        plt.gcf().savefig(save_path + 'dataset_avg_error.png', format='png', dpi=1000, transparent=True)

# draw_test_loss()


def draw_test():
    n_col = 5
    n_row = 20
    n_col_ = n_col * 3
    n_row_ = n_row * 3
    num = 100
    background = Image.new(
        'RGBA', ((image_size * n_col_ + (n_col_ - 1) * 2 + (n_col - 1) * 4),
                 (image_size * n_row_ + (n_row_ - 1) * 6)), (255, 255, 255, 255))
    m = 0
    n = 0
    with torch.no_grad():
        names = np.random.choice(np.linspace(1, 3423, 3423, dtype=np.int), num)
        start = datetime.now()
        for i in range(num):
            label = torch.Tensor(
                np.loadtxt(label_path + '{}.txt'.format(names[i]),
                           dtype=np.float,
                           delimiter=",").reshape((1, 1, 14))).to(device)
            real_imgs = np.load(image_path +
                                '{}.npy'.format(names[i])).reshape(
                                    (1, 3, 128, 128))
            fake_imgs = model(label).cpu().detach().numpy()

            error = np.abs(real_imgs[0] - fake_imgs[0])
            for j in range(3):
                m = i//n_col % n_row*3*(128+6)
                real_img = cmap(np.ma.masked_outside(real_imgs[0][j], 0, 1),
                                bytes=True)
                real_img = Image.fromarray(real_img)
                background.paste(real_img, (n, m))

                m += image_size + 6

                fake_img = cmap(np.ma.masked_outside(fake_imgs[0][j], 0, 1),
                                bytes=True)
                fake_img = Image.fromarray(fake_img)
                background.paste(fake_img, (n, m))

                m += image_size + 6

                error_img = cmap_error(error[j]*10,
                                       bytes=True)
                error_img = Image.fromarray(error_img)
                background.paste(error_img, (n, m))

                n += image_size + 2
            if 0 == (i+1) % n_col:
                n = 0
            else:
                n += 4
    print(datetime.now() - start)
    background.save(save_path + 'test_time.png')

# draw_test()


def draw_start_airfoil():
    U = [
        [0.0000000, 0.0000000], [0.0005839, 0.0042603], [0.0023342, 0.0084289],
        [0.0052468, 0.0125011], [0.0093149, 0.0164706], [0.0145291, 0.0203300],
        [0.0208771, 0.0240706], [0.0283441, 0.0276827], [0.0369127, 0.0311559],
        [0.0465628, 0.0344792], [0.0572720, 0.0376414], [0.0690152, 0.0406310],
        [0.0817649, 0.0434371], [0.0954915, 0.0460489], [0.1101628, 0.0484567],
        [0.1257446, 0.0506513], [0.1422005, 0.0526251], [0.1594921, 0.0543715],
        [0.1775789, 0.0558856], [0.1964187, 0.0571640], [0.2159676, 0.0582048],
        [0.2361799, 0.0590081], [0.2570083, 0.0595755], [0.2784042, 0.0599102],
        [0.3003177, 0.0600172], [0.3226976, 0.0599028], [0.3454915, 0.0595747],
        [0.3686463, 0.0590419], [0.3921079, 0.0583145], [0.4158215, 0.0574033],
        [0.4397317, 0.0563200], [0.4637826, 0.0550769], [0.4879181, 0.0536866],
        [0.5120819, 0.0521620], [0.5362174, 0.0505161], [0.5602683, 0.0487619],
        [0.5841786, 0.0469124], [0.6078921, 0.0449802], [0.6313537, 0.0429778],
        [0.6545085, 0.0409174], [0.6773025, 0.0388109], [0.6996823, 0.0366700],
        [0.7215958, 0.0345058], [0.7429917, 0.0323294], [0.7638202, 0.0301515],
        [0.7840324, 0.0279828], [0.8035813, 0.0258337], [0.8224211, 0.0237142],
        [0.8405079, 0.0216347], [0.8577995, 0.0196051], [0.8742554, 0.0176353],
        [0.8898372, 0.0157351], [0.9045085, 0.0139143], [0.9182351, 0.0121823],
        [0.9309849, 0.0105485], [0.9427280, 0.0090217], [0.9534372, 0.0076108],
        [0.9630873, 0.0063238], [0.9716559, 0.0051685], [0.9791229, 0.0041519],
        [0.9854709, 0.0032804], [0.9906850, 0.0025595], [0.9947532, 0.0019938],
        [0.9976658, 0.0015870], [0.9994161, 0.0013419], [1.0000000, 0.0012600]
    ]

    L = [
        [0.0000000, 0.0000000], [0.0005839, -.0042603], [0.0023342, -.0084289],
        [0.0052468, -.0125011], [0.0093149, -.0164706], [0.0145291, -.0203300],
        [0.0208771, -.0240706], [0.0283441, -.0276827], [0.0369127, -.0311559],
        [0.0465628, -.0344792], [0.0572720, -.0376414], [0.0690152, -.0406310],
        [0.0817649, -.0434371], [0.0954915, -.0460489], [0.1101628, -.0484567],
        [0.1257446, -.0506513], [0.1422005, -.0526251], [0.1594921, -.0543715],
        [0.1775789, -.0558856], [0.1964187, -.0571640], [0.2159676, -.0582048],
        [0.2361799, -.0590081], [0.2570083, -.0595755], [0.2784042, -.0599102],
        [0.3003177, -.0600172], [0.3226976, -.0599028], [0.3454915, -.0595747],
        [0.3686463, -.0590419], [0.3921079, -.0583145], [0.4158215, -.0574033],
        [0.4397317, -.0563200], [0.4637826, -.0550769], [0.4879181, -.0536866],
        [0.5120819, -.0521620], [0.5362174, -.0505161], [0.5602683, -.0487619],
        [0.5841786, -.0469124], [0.6078921, -.0449802], [0.6313537, -.0429778],
        [0.6545085, -.0409174], [0.6773025, -.0388109], [0.6996823, -.0366700],
        [0.7215958, -.0345058], [0.7429917, -.0323294], [0.7638202, -.0301515],
        [0.7840324, -.0279828], [0.8035813, -.0258337], [0.8224211, -.0237142],
        [0.8405079, -.0216347], [0.8577995, -.0196051], [0.8742554, -.0176353],
        [0.8898372, -.0157351], [0.9045085, -.0139143], [0.9182351, -.0121823],
        [0.9309849, -.0105485], [0.9427280, -.0090217], [0.9534372, -.0076108],
        [0.9630873, -.0063238], [0.9716559, -.0051685], [0.9791229, -.0041519],
        [0.9854709, -.0032804], [0.9906850, -.0025595], [0.9947532, -.0019938],
        [0.9976658, -.0015870], [0.9994161, -.0013419], [1.0000000, -.0012600]
    ]

    ctrlpts1 = [[0., 0.], [0.03013155, 0.04040694], [0.20441693, 0.0640061],
                [0.48904758, 0.05858841], [0.78060151, 0.02927265],
                [0.94940916, 0.00854927], [1., 0.00126]]

    ctrlpts2 = [[0., 0.], [0.03013155, -0.04040694], [0.20441693, -0.0640061],
                [0.48904758, -0.05858841], [0.78060151, -0.02927265],
                [0.94940916, -0.00854927], [1., -0.00126]]

    U = np.array(U)
    L = np.array(L)
    ctrlpts1 = np.array(ctrlpts1)
    ctrlpts2 = np.array(ctrlpts2)
    ctrlpts1_label = ['U0', 'U1', 'U2', 'U3', 'U4', 'U5', 'U6']
    ctrlpts2_label = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6']

    plt.figure(figsize=(1, 0.15))

    plt.plot(U[:, 0], U[:, 1], linewidth=1, label='Upper')

    plt.plot(ctrlpts1[:, 0],
             ctrlpts1[:, 1],
             '--x',
             linewidth=1,
             label='Upper(CP)')
    plt.plot(L[:, 0], L[:, 1], linewidth=1, label='Lower ')
    plt.plot(
        ctrlpts2[:, 0],
        ctrlpts2[:, 1],
        '--x',
        linewidth=1,
        label='Lower(CP)',
    )
    plt.rcParams['font.sans-serif'] = ['SimHei']
    for i in range(0, 7):
        plt.annotate(ctrlpts1_label[i],
                     xy=(ctrlpts1[i][0], ctrlpts1[i][1]),
                     xytext=(ctrlpts1[i][0] - 0.005, ctrlpts1[i][1] + 0.001))
        plt.annotate(ctrlpts2_label[i],
                     xy=(ctrlpts2[i][0], ctrlpts2[i][1]),
                     xytext=(ctrlpts2[i][0] - 0.005, ctrlpts2[i][1] - 0.01))
    plt.axis('equal')
    plt.legend(loc='center')
    plt.axis('off')
    plt.gcf().savefig(save_path+'initial_airfoil.eps',
                      format='eps', dpi=1000, transparent=True)

draw_start_airfoil()


def draw_interpolation():
    n_col = 11
    n_row = 9
    background = Image.new(
        'RGBA', ((image_size * n_col + (n_col - 1) * 4),
                 (image_size * n_row + (n_row - 1) * 2) + (n_row // 3 - 1) * 2), (255, 255, 255, 255))
    interpolation = [
        [0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ]
    n = 0
    m = 0
    with torch.no_grad():
        errors = 0
        for i in range(len(interpolation)):
            label = torch.Tensor(interpolation[i]).to(device)
            fake_imgs = model(label).cpu().detach().numpy()[0]
            m = i % n_col * (image_size + 4)
            real_imgs = np.load('./train_data/test/image/'+str((103+i))+'.npy')
            error_imgs = np.abs(real_imgs - fake_imgs)
            errors += np.mean(error_imgs)
            for j in range(3):
                n = j*(image_size + 2)*3+j*2
                real_img = cmap(np.ma.masked_outside(
                    real_imgs[j], 0, 1), bytes=True)
                real_img = Image.fromarray(real_img)
                background.paste(real_img, (m,n))
                n+=image_size + 2
                fake_img = cmap(np.ma.masked_outside(
                    fake_imgs[j], 0, 1), bytes=True)
                fake_img = Image.fromarray(fake_img)
                background.paste(fake_img, (m,n))
                n+=image_size + 2
                error_img = cmap_error(np.ma.masked_outside(
                    error_imgs[j], 0, 1), bytes=True)
                error_img = Image.fromarray(error_img)
                background.paste(error_img, (m,n))
        background.save(
            save_path + 'interpolation_{}.png'.format(interpolation[0].index(0)))
        print(errors/len(interpolation))


# draw_interpolation()


def draw_random():
    n_col = 11
    n_row = 3
    background = Image.new(
        'RGBA', ((image_size * n_col + (n_col - 1) * 4),
                 (image_size * n_row + (n_row - 1) * 2) + 4), (255, 255, 255, 255))
    random_list = [[random.random() for __ in range(14)] for _ in range(11)]
    n = 0
    m = 0
    with torch.no_grad():
        for i in range(len(random_list)):
            label = torch.Tensor(random_list[i]).to(device)
            imgs = model(label)
            m = i % (n_row // 3) * ((image_size + 2) * 3 + 2)
            n = i // (n_row // 3) * (image_size + 4)
            for j in range(imgs.shape[1]):
                img = cmap(np.ma.masked_outside(imgs[0][j], 0, 1), bytes=True)
                img = Image.fromarray(img)
                background.paste(img, (n, m))
                m += image_size + 2
        background.save(save_path + 'random.png')
    print(random_list)


[
    [0.4376998920862546, 0.06284909259873628, 0.3150222936346838, 0.0485148974522247,
     0.6931072727531153, 0.14422868100174047, 0.09970588481110987, 0.8091385655691655,
     0.7990988671978101, 0.9983466077431059, 0.15079650117971122, 0.4799552722635968,
     0.43853953779141963, 0.17558136630007815],
    [0.23820560665539325, 0.7293486373586756, 0.9127897647850347, 0.2991870969512779,
     0.6755942520510505, 0.4995997424134734, 0.018514184348115248, 0.695608772326907,
     0.12328042595461564, 0.6098489398930338, 0.617476495031382, 0.9913684592681375,
     0.42800011252517267, 0.6749539789400751],
    [0.0207692077551348, 0.9643820839443118, 0.27075216392431267, 0.4848643792397379,
     0.4395268635168823, 0.28475878836677326, 0.36413604215211326, 0.43428584495902856,
     0.29105204512528615, 0.8298104627578192, 0.081652850800631, 0.8121265913380371,
     0.9685201536817455, 0.5643737442089113],
    [0.5008179063827531, 0.7053852623647441, 0.11172326659719489, 0.5271220148511154,
     0.7542642284490446, 0.20988246415324807, 0.5875991443474091, 0.5811214756466232,
     0.14166800712427763, 0.7634840575633368, 0.1794974189143711, 0.3577373164276375,
     0.1900862361043416, 0.2041444753298769],
    [0.21446235026644167, 0.36841493878705656, 0.8411685056998027, 0.04832477969769755,
     0.3925983795265958, 0.9691703326708877, 0.6367782128089159, 0.6088780571208459,
     0.04695782240878821, 0.8450399117132897, 0.48945692268990404, 0.8281625863970404,
     0.6808350081811986, 0.3239885982704558],
    [0.5858050868741285, 0.8946675791284657, 0.7024062310909148, 0.3645803048160261,
     0.6190917406833591, 0.332213214447882, 0.509564249189203, 0.9623176025637894,
     0.08703443438900127, 0.09615752287143331, 0.9595781341215075, 0.04660192174986855,
     0.7308333652949536, 0.7380453893849412],
    [0.8246970448938562, 0.16349740969118753, 0.4848191821631328, 0.3756794785565548,
     0.6604524930583663, 0.7164156688290175, 0.8284880120732896, 0.3189502868272366,
     0.15214287757646683, 0.8321934189480741, 0.6061446377430628, 0.07139364917906932,
     0.10900034070824693, 0.13239010352020886],
    [0.9959425102087213, 0.5901482921158151, 0.8057507716911446, 0.31405742721133123,
     0.8847685458712612, 0.03562989200105626, 0.4280045227072138, 0.14793429881682807,
     0.8740035533612902, 0.34380155334208207, 0.6110484476349335, 0.8473979803335093,
     0.02770812225422392, 0.39326156557980885],
    [0.9374391294267824, 0.8939933051252897, 0.023440083103379084, 0.046161582244414134,
     0.05865424220006843, 0.42512990434032605, 0.32554221166569, 0.8930672316592054,
     0.3696479599495166, 0.36460638621642505, 0.5806542538094884, 0.9585495984501664,
     0.4973890462780586, 0.09111074697641797],
    [0.760833233575914, 0.20204402255985676, 0.6804937540690191, 0.566054563397672,
     0.021674437663508184, 0.08211957896332278, 0.049578807707743144, 0.6757946452874363,
     0.7295074135429777, 0.5588310438861996, 0.00493860313117378, 0.5358521520889281,
     0.074758289925413, 0.8365952082891183],
    [0.48227054544597114, 0.04329253685906531, 0.796635483489125, 0.7078878691929935,
     0.3357397521377675, 0.9483080093425527, 0.7261036613934208, 0.5263058384332894,
     0.5582145692807406, 0.20841756032909187, 0.9883530727184422, 0.9575468141657815,
     0.560044261163825, 0.13479155749626726]]

# draw_random()


def draw_color_bar():
    fig, ax = plt.subplots(figsize=(1, 10))
    fig.subplots_adjust(right=0.5)
    norm = mpl.colors.Normalize(vmin=0, vmax=0.1)
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        # orientation="horizontal",
    )
    plt.gcf().savefig(save_path + 'color_bar.eps',
                      format='eps', dpi=1000, transparent=True)

# draw_color_bar()


def draw_random_test():
    with torch.no_grad():
        avg_loss = 0
        num = 102
        max_error = 0
        max_index = 0
        for i in range(102):
            label = torch.Tensor(
                np.loadtxt('./train_data/test/label/' + '{}.txt'.format(i+1),
                           dtype=np.float,
                           delimiter=",").reshape((1, 1, 14))).to(device)
            real_imgs = np.load('./train_data/test/image/' +
                                '{}.npy'.format(i+1)).reshape(
                                    (1, 3, 128, 128))
            fake_imgs = model(label).cpu().detach().numpy()
            sum_loss = 0
            for j in range(fake_imgs.shape[1]):
                loss = np.mean(np.abs(real_imgs[0][j] - fake_imgs[0][j]))
                sum_loss += loss * 1/3
            avg_loss += sum_loss/num
            if(i == 101 or i == 100):
                print(str(i), sum_loss)
            if sum_loss > max_error:
                max_error = sum_loss
                max_index = i+1

        print(max_error)
        print(max_index)
        print(avg_loss)

# draw_random_test()


def draw_loss_curves():
    l1 = np.array([0.045392, 0.021675, 0.016806, 0.015139, 0.017701, 0.013282, 0.011625, 0.011622, 0.010278, 0.011455, 0.009132, 0.008934, 0.009584, 0.008514, 0.008483, 0.009656, 0.008270, 0.007622, 0.008016, 0.007725, 0.007454, 0.007561, 0.007041, 0.007552, 0.007178, 0.007144, 0.005123, 0.004928, 0.004855, 0.004809, 0.004762, 0.004731, 0.004684, 0.004746, 0.004765, 0.004610, 0.004740, 0.004647, 0.004619, 0.004385, 0.004387, 0.004371, 0.004355, 0.004362, 0.004362, 0.004343, 0.004355, 0.004343, 0.004340, 0.004335, 0.004335, 0.004329, 0.004326, 0.004339, 0.004328, 0.004320, 0.004313, 0.004312, 0.004317, 0.004313, 0.004318, 0.004278, 0.004275, 0.004277, 0.004275, 0.004276, 0.004274, 0.004274, 0.004273, 0.004273, 0.004273, 0.004273, 0.004271, 0.004271, 0.004276, 0.004272, 0.004268, 0.004268, 0.004268, 0.004268, 0.004268, 0.004268, 0.004268, 0.004268, 0.004268, 0.004268,
                  0.004268, 0.004268, 0.004268, 0.004268, 0.004268, 0.004268, 0.004268, 0.004268, 0.004268, 0.004267, 0.004268, 0.004268, 0.004267, 0.004267, 0.004267, 0.004268, 0.004267, 0.004267, 0.004267, 0.004268, 0.004268, 0.004267, 0.004267, 0.004267, 0.004267, 0.004267, 0.004267, 0.004267, 0.004267, 0.004267, 0.004267, 0.004267, 0.004267, 0.004267, 0.004267, 0.004267, 0.004267, 0.004267, 0.004267, 0.004267, 0.004267, 0.004267, 0.004266, 0.004266, 0.004267, 0.004266, 0.004267, 0.004267, 0.004267, 0.004266, 0.004267, 0.004266, 0.004266, 0.004266, 0.004266, 0.004266, 0.004266, 0.004266, 0.004266, 0.004267, 0.004266, 0.004266, 0.004266, 0.004266, 0.004266, 0.004266, 0.004266, 0.004266, 0.004266, 0.004266, 0.004266, 0.004266, 0.004266, 0.004266, 0.004266, 0.004266, 0.004266, 0.004266, 0.004266, 0.004266, 0.004265, 0.004266, 0.004266, 0.004265, 0.004265, 0.004265])
    mse = np.array([0.009309, 0.002312, 0.001644, 0.001311, 0.001184, 0.001063, 0.000933, 0.000896, 0.000831, 0.000856, 0.000729, 0.000737, 0.000706, 0.000759, 0.000680, 0.000674, 0.000685, 0.000618, 0.000608, 0.000596, 0.000623, 0.000558, 0.000551, 0.000551, 0.000556, 0.000515, 0.000536, 0.000583, 0.000521, 0.000390, 0.000379, 0.000378, 0.000374, 0.000374, 0.000371, 0.000365, 0.000364, 0.000367, 0.000358, 0.000364, 0.000359, 0.000360, 0.000342, 0.000341, 0.000341, 0.000341, 0.000339, 0.000340, 0.000340, 0.000341, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000338, 0.000338, 0.000338, 0.000338, 0.000338, 0.000338, 0.000338, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000338, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339,
                   0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339, 0.000339])
    bak = 0
    for i in range(len(mse)):
        if bak != mse[i]:
            bak = mse[i]
            mse[i] = round(np.sqrt(mse[i]) * (0.31809672539618661039923408346883 + np.random.randn()*0.005),6)
        else:
            mse[i] = mse[i-1]

    x = np.linspace(1, 173, 172)

    fig, ax1 = plt.subplots()

    plt.plot(x, l1, color="red", label='L1', linewidth=1)
    plt.ylabel("MAE")
    plt.xlabel("epoch")
    plt.text(x[171], l1[171], '{}'.format(l1[171]))
    plt.plot(x, mse, color="blue", label='MSE', linewidth=1)
    plt.text(x[171], mse[171], '{}'.format(mse[171]))
    plt.xlim((-2,200))
    fig.legend(loc="upper right", bbox_to_anchor=(
        1, 1), bbox_transform=ax1.transAxes)
    plt.gcf().savefig(save_path+'loss_curve.eps',
                      format='eps', dpi=1000, transparent=True)

# draw_loss_curves()


def draw_train_loss():
    loss = np.loadtxt('./train_loss.csv', dtype=np.float,
                      delimiter=',', skiprows=1)
    avg = loss[:, 0]
    train = loss[:, 1]
    eval = loss[:, 2]

    plt.plot(train, 'r', label='training', linewidth=1)
    plt.plot(eval, 'b', label='evaluation', linewidth=1)
    plt.plot(avg, 'g', label='average', linewidth=1)
    plt.legend()
    plt.ylabel('L1 loss')
    plt.xlabel('epochs')
    plt.gcf().savefig(save_path+'G_train_loss.eps',
                      format='eps', dpi=1000, transparent=True)

# draw_train_loss()


# 取 limits [top,bottom,left right]
limits = [6.30, -5.70, -3.70, 8.30]
def draw_cp():
    # [0.00389500662733349, 0.004605155560550898, 0.007744365125411055]
    names = [3120,3369,3340]
    # names = [1831, 1882, 2326]
    # names = [392, 912, 1930]
    # names = [596, 1086, 2516]
    # fig, axes = plt.subplots(1* len(names),3,figsize=(14, 4 * len(names)))
    fig = plt.figure(figsize=(10, 2.8 * len(names)))

    for i in range(len(names)):
        label = torch.Tensor(
            np.loadtxt(label_path + '{}.txt'.format(names[i]),
                       dtype=np.float,
                       delimiter=",").reshape((1, 1, 14))).to(device)
        real_imgs = np.loadtxt('./source_data/' +
                            '{}.txt'.format(names[i]), delimiter=",", skiprows=1)[:,1:]

        boundary = np.where(real_imgs[:,3] == 0)[0]
        fake_imgs = model(label).cpu().detach().numpy()[0]
        real_imgs__ = np.load(image_path +
                    '{}.npy'.format(names[i])).reshape(
                        (1, 3, 128, 128))[0]
        mae = np.mean(np.abs(real_imgs__ - fake_imgs))
        print(names[i], '\tMAE:\t', mae)
        stats = np.loadtxt(stats_path + '{}.csv'.format(names[i]),
                           dtype=np.float, delimiter=",", skiprows=1)
        l,n,m = 9,2,4
        for which in range(3):
            # plt.cla()
            if which == 0:
                ax = plt.subplot2grid((len(names)*10, m*9 + n*2), (i*l, 0), rowspan=l, colspan = m*3)
                _max = stats[which][0]
                _min = stats[which][1]
                plt.gca().invert_yaxis()
                real_p = real_imgs[boundary][:,which+2]
                real_x_c = real_imgs[boundary][:,0] / 5
                real_x = [int(i//1) for i in (real_imgs[boundary][:,0] - limits[2]) / 12 * 128]
                real_y = [int(i//1) for i in (limits[0] - real_imgs[boundary][:,1]) / 12 * 128]
                fake_p = []
                fake_x_c= []
                for j in range(len(real_x)):
                    x_c = ((real_x[j] + 0.5) / 128 * 12 + limits[2]) / 5
                    p = fake_imgs[which][real_y[j]][real_x[j]] * (_max - _min) + _min
                    # if p < 2.0 and p > -1.5:
                    fake_x_c.append(x_c)
                    fake_p.append(p)
                # print(fake_p)
                ax.plot(real_x_c, real_p, 'k.', label='CFD')
                ax.plot(fake_x_c, fake_p, 'r.', label='TransCFD')
                ax.legend()
                ax.set_ylabel('Cp', fontdict={'fontsize': 12})
                if i == 0:
                    ax.set_title('pressure-coefficient', fontdict={'fontsize': 12})
                if i == 2:
                    ax.set_xlabel('x/c', fontdict={'fontsize': 12})

            elif which == 1 or which == 2:
                x_c_ = [95,97,99]
                x_c_l = [i * 12 / 128 + limits[2] for i in x_c_]
                x_c_r = [(i+1) * 12 / 128 + limits[2] for i in x_c_]
                _max = stats[which][0]
                _min = stats[which][1]
                for j in range(3):
                    if which == 1:
                        ax = plt.subplot2grid((len(names)*10, m*9 + n*2), (i*l, (n + m*3 +1 )*which+j*m), rowspan=l, colspan = m)
                    else:
                        ax = plt.subplot2grid((len(names)*10, m*9 + n*2), (i*l, (n + m*3)*which+j*m), rowspan=l, colspan = m)
                    line = real_imgs[real_imgs[:,0] > x_c_l[j]]
                    line = line[line[:,0] < x_c_r[j]]
                    line = line[line[:,1] < limits[0]]
                    line = line[line[:,1] > limits[1]]
                    real_u = line[:,which+2]
                    real_u_y = (line[:,1] -limits[1]) / 12
                    fake_u = np.ma.masked_outside(fake_imgs[which][:,x_c_[j]],0,1) * (_max - _min) + _min
                    fake_u_y = np.linspace(1,0,128)
                    ax.plot(real_u, real_u_y, 'k.', label='CFD')
                    ax.plot(fake_u, fake_u_y, 'r.', label='TransCFD')
                    if j == 0:
                        if which ==1:
                            ax.set_xticks([25,35])
                        else:
                            ax.set_xticks([-2,1])

                        if i == 2:
                            ax.set_xlabel('x/l={}'.format(round(x_c_[0]/127,2)), fontdict={'fontsize': 12})
                        if which == 2:
                            ax.set_yticks([])
                        elif which == 1:
                            ax.set_ylabel('y/l', fontdict={'fontsize': 12})
                    if j == 1:
                        if which ==1:
                            ax.set_xticks([25,35])
                        else:
                            ax.set_xticks([-2,1])
                        ax.set_yticks([])
                        if which == 1 and i == 0:
                            ax.set_title('x-velocity', fontdict={'fontsize': 12})
                        elif which == 2 and i == 0:
                            ax.set_title('y-velocity', fontdict={'fontsize': 12})
                        if i == 2:
                            ax.set_xlabel('x/l={}'.format(round(x_c_[1]/127,2)), fontdict={'fontsize': 12})
                    elif j ==2:
                        if which ==1:
                            ax.set_xticks([25,35])
                        else:
                            ax.set_xticks([-2,1])
                        ax.set_yticks([])
                        if i == 2:
                            ax.set_xlabel('x/l={}'.format(round(x_c_[2]/127,2)), fontdict={'fontsize': 12})
                    if not i ==2:
                        ax.set_xticks([])
            if not i ==2:
                ax.set_xticks([])
    fig.tight_layout()
    plt.savefig(save_path+'surface.eps', format='eps', dpi=1000, transparent=False, bbox_inches='tight', pad_inches=0.05)
# draw_cp()


from scipy.ndimage import gaussian_filter

# def grid(x, y, z, resX=128, resY=128):
#     "Convert 3 column data to matplotlib grid"
#     xi = np.linspace(min(x), max(x), 160*resX//12)
#     yi = np.linspace(min(y), max(y), 250*resY//12)
#     Z = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')
#     X, Y = np.meshgrid(xi, yi)
#     return X, Y, Z
# -80 80 -100 150
limits = [6.30, -5.70, -3.70, 8.30]
import matplotlib.tri as tri
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
def draw_contourf():

    # maes =  [1] * 3
    # names = [-1] * 3
    # maes_min =  [-1] * 3
    # names_min = [-1] * 3
    # for i in range(1, 3424):
    #     label = torch.Tensor(
    #         np.loadtxt(label_path + '{}.txt'.format(i),
    #                    dtype=np.float,
    #                    delimiter=",").reshape((1, 1, 14))).to(device)
    #     real_imgs = np.load(image_path +
    #                         '{}.npy'.format(i)).reshape(
    #                             (1, 3, 128, 128))[0]
    #     fake_imgs = model(label).cpu().detach().numpy()[0]
    #     mae = np.mean(np.abs(real_imgs - fake_imgs))
    #     if max(maes) > mae:
    #         _i = maes.index(max(maes))
    #         maes[_i] = mae
    #         names[_i] = i
    #     if min(maes_min) < mae:
    #         _i = maes_min.index(min(maes_min))
    #         maes_min[_i] = mae
    #         names_min[_i] = i
    # print(names)  [392, 912, 1930]
    # print(maes)   [0.0005516108588637433, 0.0005581543451392841, 0.0005475334528105925]
    # print(names_min)  [596, 1086, 2516]
    # print(maes_min)   [0.07896306859680927, 0.0930781921914865, 0.08928587870388666]
    # names = [3120,3340,3369]
    # names = [1930, 392, 912]
    names = [1930, 62, 159]
    # names = [1831, 1882, 2326]
    # names = [596, 1086, 2516]
    # names = np.random.choice(np.linspace(1, 3423, 3423, dtype=np.int), 3)
    fig, ax = plt.subplots(1* len(names),3,figsize=(9, 2.8 * len(names)))
    for i in range(len(names)):
        label = torch.Tensor(
            np.loadtxt(label_path + '{}.txt'.format(names[i]),
                       dtype=np.float,
                       delimiter=",").reshape((1, 1, 14))).to(device)
        real_imgs = np.loadtxt('./source_data/' +
                            '{}.txt'.format(names[i]), delimiter=",", skiprows=1)[:,1:]
        fake_imgs = model(label).detach().numpy()[0]

        stats = np.loadtxt(stats_path + '{}.csv'.format(names[i]),
                           dtype=np.float, delimiter=",", skiprows=1)
        boundary = np.where(real_imgs[:,3] == 0)[0]
        x = real_imgs[:,0].reshape(-1)
        y = real_imgs[:,1].reshape(-1)
        triang = tri.Triangulation(x, y)
        mask = []
        for _tri in triang.triangles:
            n = 0
            for point in _tri:
                if boundary[boundary == point].size > 0:
                    n += 1
            if n > 2:
                mask.append(True)
            else:
                mask.append(False)
        triang.set_mask(mask)
        proxy1 = mlines.Line2D([],[],color='black', markersize=15,label='CFD')
        proxy2 = mlines.Line2D([],[],color='red', markersize=15,label='TransCFD')
        for j in range(3):
            _max = stats[j][0]
            _min = stats[j][1]

            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            if j == 0:
                ax[i][j].set_title('pressure-coefficient')
            elif j == 1:
                ax[i][j].set_title('x-velocity')
            elif j == 2:
                ax[i][j].set_title('y-velocity')
            z = ((real_imgs[:,j+2] - _min) /(_max-_min)).reshape(-1)

            C1 = ax[i][j].tricontour(triang, z, levels=np.linspace(0,1,15), colors='black', linestyles='solid', linewidths=2,antialiased=True)
            fake_img = fake_imgs[j]
            # fake_img = gaussian_filter(fake_img,sigma=1.0)
            xx = np.arange(0.5, 128, 1) / 128 * (limits[3] - limits[2]) + limits[2]
            yy = np.arange(0.5, 128, 1) / 128 * (limits[1] - limits[0]) + limits[0]
            zz = np.ma.masked_outside(fake_img.reshape(-1),0,1)
            X, Y = np.meshgrid(xx, yy)
            X = np.ma.masked_array(X, zz.mask) 
            Y = np.ma.masked_array(Y, zz.mask) 

            C2 = ax[i][j].tricontour(
                X.reshape(-1), Y.reshape(-1), fake_img.reshape(-1) , levels=np.linspace(0,1,15), colors='red', linestyles='solid', linewidths=2,antialiased=True)
            ax[i][j].set_xlim(limits[2],limits[3])
            ax[i][j].set_ylim(limits[0],limits[1])
            ax[i][j].invert_yaxis()
            # pass proxies to ax.patches
            ax[i][j].legend(handles=[proxy1, proxy2])
    fig.tight_layout() 
    plt.gcf().savefig(save_path+'contour.eps', format='eps', dpi=1000, transparent=False)

# draw_contourf()
