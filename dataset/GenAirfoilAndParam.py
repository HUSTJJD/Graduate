from utilty import DoE_LHS, naca4, cst, PlotAirfoils
from multiprocessing.pool import Pool
import numpy as np
import os

os.chdir(os.path.dirname(__file__))


def GenAirfoil():
    N = 1000
    os.mkdir('./' + dirpath + '/dat') if not os.path.exists('./' +
                                                            dirpath + '/dat') else None

    # name_value = ["m", "p", "t", "mach", "aoa"]
    # arr_limit = np.array([[0.0, 0.0, 0.05, mach_min, aoa_min],
    #                       [0.07, 0.5, 0.25, mach_max, aoa_max]]).T

    name_value = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6',
                  'l1', 'l2', 'l3', 'l4', 'l5', 'l6']

    arr_limit = np.array([[0.15, 0.15, 0.15, 0.15, 0.15, 0.10, -0.20, -0.20, -0.15, -0.15, -0.10, -0.05],
                          [0.35, 0.45, 0.45, 0.45, 0.35, 0.30, -0.10,  0.00,  0.02,  0.02,  0.05,  0.10]]).T
    # arr_limit = np.array([[1.00, 1.00, 1.00, 1.00, 1.00, 1.00, -1.00, -1.00, -1.00, -1.00, -1.00, -1.00],
    #                       [0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05]]).T

    # 最大弯度, 最大弯度位置, 最大厚度, 马赫数, 攻角
    lhs = DoE_LHS(N=N, bounds=arr_limit, name_value=name_value)
    data = lhs.parameterArray.tolist()
    lhs.write_to_csv('./'+dirpath+'Airfoil.csv')

    thread_pool = Pool()
    for i in range(len(data)):
        temp = [dirpath, str(i)]
        temp.extend(data[i])
        # thread_pool.apply_async(naca4, (temp,))
        thread_pool.apply_async(cst, (temp,))

    thread_pool.close()
    thread_pool.join()





def GenParam():
    N = 10
    mach_max, mach_min = 0.5, 0.1
    aoa_max, aoa_min = 5, 0
    name_value = ['mach', 'aoa']
    arr_limit = np.array([[mach_min, aoa_min], [mach_max, aoa_max]]).T
    # 最大弯度, 最大弯度位置, 最大厚度, 马赫数, 攻角
    lhs = DoE_LHS(N=N, bounds=arr_limit, name_value=name_value)
    lhs.write_to_csv('./'+dirpath+'Param.csv')


dirpath = 'CST'

# PlotAirfoils(dirpath)

# GenAirfoil()

# GenParam()
