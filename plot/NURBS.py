from geomdl import NURBS, utilities
import numpy as np
import matplotlib.pyplot as plt


def draw_start_airfoil():
    U =[[0.00000000e+00, 0.00000000e+00], [1.26857795e+00, 1.78935191e+01], [5.00713910e+00, 3.55511520e+01],
        [1.11168566e+01, 5.29609676e+01], [1.95015210e+01, 7.01118461e+01], [3.00675884e+01, 8.69934327e+01],
        [4.27242239e+01, 1.03596091e+02], [5.73833404e+01, 1.19910860e+02], [7.39596340e+01, 1.35929412e+02],
        [9.23706142e+01, 1.51644007e+02], [1.12536630e+02, 1.67047459e+02], [1.34380891e+02, 1.82133092e+02],
        [1.57829486e+02, 1.96894708e+02], [1.82811395e+02, 2.11326544e+02], [2.09258499e+02, 2.25423245e+02],
        [2.37105584e+02, 2.39179826e+02], [2.66290337e+02, 2.52591645e+02], [2.96753350e+02, 2.65654365e+02],
        [3.28438103e+02, 2.78363933e+02], [3.61290955e+02, 2.90716550e+02], [3.95261125e+02, 3.02708642e+02],
        [4.30300672e+02, 3.14336840e+02], [4.66364468e+02, 3.25597950e+02], [5.03410164e+02, 3.36488941e+02],
        [5.41398160e+02, 3.47006914e+02], [5.80291565e+02, 3.57149089e+02], [6.20056150e+02, 3.66912786e+02],
        [6.60660303e+02, 3.76295407e+02], [7.02074974e+02, 3.85294424e+02], [7.44273622e+02, 3.93907362e+02],
        [7.87232151e+02, 4.02131788e+02], [8.30928845e+02, 4.09965302e+02], [8.75344299e+02, 4.17405523e+02],
        [9.20461344e+02, 4.24450088e+02], [9.66264969e+02, 4.31096636e+02], [1.01274224e+03, 4.37342813e+02],
        [1.05988220e+03, 4.43186257e+02], [1.10767581e+03, 4.48624605e+02], [1.15611580e+03, 4.53655488e+02],
        [1.20519663e+03, 4.58276529e+02], [1.25491434e+03, 4.62485348e+02], [1.30526646e+03, 4.66279563e+02],
        [1.35625188e+03, 4.69656797e+02], [1.40787077e+03, 4.72614679e+02], [1.46012440e+03, 4.75150856e+02],
        [1.51301508e+03, 4.77262998e+02], [1.56654596e+03, 4.78948812e+02], [1.62072097e+03, 4.80206047e+02],
        [1.67554460e+03, 4.81032512e+02], [1.73102183e+03, 4.81426089e+02], [1.78715793e+03, 4.81384747e+02],
        [1.84395833e+03, 4.80906561e+02], [1.90142845e+03, 4.79989725e+02], [1.95957356e+03, 4.78632581e+02],
        [2.01839859e+03, 4.76833629e+02], [2.07790799e+03, 4.74591559e+02], [2.13810551e+03, 4.71905268e+02],
        [2.19899408e+03, 4.68773888e+02], [2.26057555e+03, 4.65196814e+02], [2.32285060e+03, 4.61173729e+02],
        [2.38581845e+03, 4.56704632e+02], [2.44947674e+03, 4.51789875e+02], [2.51382129e+03, 4.46430189e+02],
        [2.57884588e+03, 4.40626721e+02], [2.64454211e+03, 4.34381065e+02], [2.71089909e+03, 4.27695302e+02],
        [2.77790332e+03, 4.20572038e+02], [2.84553838e+03, 4.13014439e+02], [2.91378477e+03, 4.05026275e+02],
        [2.98261964e+03, 3.96611958e+02], [3.05201656e+03, 3.87776591e+02], [3.12194531e+03, 3.78526007e+02],
        [3.19237158e+03, 3.68866818e+02], [3.26325679e+03, 3.58806459e+02], [3.33455778e+03, 3.48353243e+02],
        [3.40622656e+03, 3.37516403e+02], [3.47821010e+03, 3.26306151e+02], [3.55044998e+03, 3.14733726e+02],
        [3.62288221e+03, 3.02811449e+02], [3.69543689e+03, 2.90552779e+02], [3.76803794e+03, 2.77972372e+02],
        [3.84060284e+03, 2.65086137e+02], [3.91304234e+03, 2.51911296e+02], [3.98526015e+03, 2.38466446e+02],
        [4.05715265e+03, 2.24771620e+02], [4.12860858e+03, 2.10848355e+02], [4.19950878e+03, 1.96719750e+02],
        [4.26972580e+03, 1.82410542e+02], [4.33912366e+03, 1.67947165e+02], [4.40755751e+03, 1.53357826e+02],
        [4.47487329e+03, 1.38672573e+02], [4.54090740e+03, 1.23923368e+02], [4.60548641e+03, 1.09144160e+02],
        [4.66842667e+03, 9.43709602e+01], [4.72953403e+03, 7.96419195e+01], [4.78860341e+03, 6.49974048e+01],
        [4.84541857e+03, 5.04800788e+01], [4.89975163e+03, 3.61349808e+01], [4.95136280e+03, 2.20096082e+01],
        [5.00000000e+03, 8.15400000e+00]]

    L =[[0.00000000e+00,  0.00000000e+00], [2.25471293e+00, -1.03833231e+01], [8.92003974e+00, -1.99050277e+01],
        [1.98467675e+01, -2.86122589e+01], [3.48845532e+01, -3.65503645e+01], [5.38821645e+01, -4.37629282e+01],
        [7.66877148e+01, -5.02918018e+01], [1.03148893e+02, -5.61771385e+01], [1.33113190e+02, -6.14574257e+01],
        [1.66428113e+02, -6.61695185e+01], [2.02941401e+02, -7.03486731e+01], [2.42501237e+02, -7.40285803e+01],
        [2.84956440e+02, -7.72413992e+01], [3.30156672e+02, -8.00177917e+01], [3.77952623e+02, -8.23869557e+01],
        [4.28196200e+02, -8.43766604e+01], [4.80740702e+02, -8.60132798e+01], [5.35441002e+02, -8.73218281e+01],
        [5.92153710e+02, -8.83259940e+01], [6.50737339e+02, -8.90481762e+01], [7.11052464e+02, -8.95095177e+01],
        [7.72961871e+02, -8.97299421e+01], [8.36330708e+02, -8.97281884e+01], [9.01026624e+02, -8.95218468e+01],
        [9.66919905e+02, -8.91273948e+01], [1.03388361e+03, -8.85602328e+01], [1.10179368e+03, -8.78347206e+01],
        [1.17052908e+03, -8.69642135e+01], [1.23997189e+03, -8.59610989e+01], [1.31000744e+03, -8.48368331e+01],
        [1.38052438e+03, -8.36019780e+01], [1.45141481e+03, -8.22662381e+01], [1.52257435e+03, -8.08384978e+01],
        [1.59390222e+03, -7.93268587e+01], [1.66530136e+03, -7.77386771e+01], [1.73667845e+03, -7.60806019e+01],
        [1.80794403e+03, -7.43586120e+01], [1.87901253e+03, -7.25780552e+01], [1.94980236e+03, -7.07436854e+01],
        [2.02023591e+03, -6.88597016e+01], [2.09023966e+03, -6.69297865e+01], [2.15974419e+03, -6.49571450e+01],
        [2.22868421e+03, -6.29445429e+01], [2.29699861e+03, -6.08943465e+01], [2.36463047e+03, -5.88085616e+01],
        [2.43152711e+03, -5.66888727e+01], [2.49764004e+03, -5.45366828e+01], [2.56292504e+03, -5.23531531e+01],
        [2.62734214e+03, -5.01392430e+01], [2.69085559e+03, -4.78957501e+01], [2.75343389e+03, -4.56233504e+01],
        [2.81504974e+03, -4.33226391e+01], [2.87568008e+03, -4.09941705e+01], [2.93530598e+03, -3.86384995e+01],
        [2.99391271e+03, -3.62562222e+01], [3.05148961e+03, -3.38480170e+01], [3.10803012e+03, -3.14146861e+01],
        [3.16353169e+03, -2.89571966e+01], [3.21799576e+03, -2.64767225e+01], [3.27142767e+03, -2.39746862e+01],
        [3.32383663e+03, -2.14528009e+01], [3.37523563e+03, -1.89131122e+01], [3.42564137e+03, -1.63580406e+01],
        [3.47507419e+03, -1.37904243e+01], [3.52355796e+03, -1.12135615e+01], [3.57112003e+03, -8.63125314e+00],
        [3.61779111e+03, -6.04784636e+00], [3.66360515e+03, -3.46827724e+00], [3.70859928e+03, -8.98114278e-01],
        [3.75281364e+03,  1.65639807e+00], [3.79629133e+03,  4.18829575e+00], [3.83907821e+03,  6.68985128e+00],
        [3.88122284e+03,  9.15252968e+00], [3.92277631e+03,  1.15669443e+01], [3.96379209e+03,  1.39228126e+01],
        [4.00432591e+03,  1.62089113e+01], [4.04443560e+03,  1.84130320e+01], [4.08418094e+03,  2.05219361e+01],
        [4.12362348e+03,  2.25213099e+01], [4.16282637e+03,  2.43957194e+01], [4.20185423e+03,  2.61285646e+01],
        [4.24077290e+03,  2.77020344e+01], [4.27964934e+03,  2.90970605e+01], [4.31855136e+03,  3.02932720e+01],
        [4.35754748e+03,  3.12689485e+01], [4.39670670e+03,  3.20009748e+01], [4.43609832e+03,  3.24647938e+01],
        [4.47579169e+03,  3.26343602e+01], [4.51585603e+03,  3.24820939e+01], [4.55636018e+03,  3.19788328e+01],
        [4.59737239e+03,  3.10937857e+01], [4.63896009e+03,  2.97944852e+01], [4.68118960e+03,  2.80467406e+01],
        [4.72412598e+03,  2.58145893e+01], [4.76783268e+03,  2.30602501e+01], [4.81237135e+03,  1.97440746e+01],
        [4.85780155e+03,  1.58244995e+01], [4.90418048e+03,  1.12579979e+01], [4.95156274e+03,  5.99903113e+00],
        [5.00000000e+03,  0.00000000e+00]]

    ctrlpts1 = [[0, 0], [0, 297.12], [840, 521.33], [1689.9, 565.63],
                [2389.18, 565.63], [4223.46, 234.26], [5000, 8.154]]

    ctrlpts2 = [[0, 0], [0, -178.7], [1489.29, -59.96], [3278.56, -47.24],
                [3744.23, -47.24], [4191.81, 105.36], [5000, 0]]

    U = np.array(U)/5000
    L = np.array(L)/5000
    ctrlpts1 = np.array(ctrlpts1)/5000
    ctrlpts2 = np.array(ctrlpts2)/5000
    ctrlpts1_label = ['U0', 'U1', 'U2', 'U3', 'U4', 'U5', 'U6']
    ctrlpts2_label = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6']

    plt.figure(figsize=(14, 1.5))

    plt.plot(U[:, 0], U[:, 1], 'k--|', markersize=4,
             linewidth=1, label='Upper')

    plt.plot(ctrlpts1[:, 0],
             ctrlpts1[:, 1],
             'k--v',
             linewidth=1,
             markersize=4,
             label='Upper(CP)')
    plt.plot(L[:, 0], L[:, 1], 'k--_', markersize=4,
             linewidth=1, label='Lower ')
    plt.plot(
        ctrlpts2[:, 0],
        ctrlpts2[:, 1],
        'k--^',
        linewidth=1,
        markersize=4,
        label='Lower(CP)',
    )
    plt.rcParams['font.sans-serif'] = ['SimHei']
    for i in range(0, 7):
        plt.annotate(ctrlpts1_label[i],
                     xy=(ctrlpts1[i][0], ctrlpts1[i][1]),
                     xytext=(ctrlpts1[i][0] - 0.005, ctrlpts1[i][1] + 0.01))
        plt.annotate(ctrlpts2_label[i],
                     xy=(ctrlpts2[i][0], ctrlpts2[i][1]),
                     xytext=(ctrlpts2[i][0] - 0.005, ctrlpts2[i][1] - 0.03))
    plt.axis('equal')
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.gcf().savefig('initial_airfoil.png',
                      format='png', dpi=600, transparent=True)


draw_start_airfoil()
