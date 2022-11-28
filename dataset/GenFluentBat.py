from shutil import copy
import numpy as np
import re
import os
import math

os.chdir(os.path.dirname(__file__))

TYPE = 'CST'

jou = 'template.jou'

fluent_bat = 'fluent.bat'

fluent_shell = 'fluent.sh'

yh_shell = 'yhfluent.sh'

get_jou = False

create_bat = False

windows_path = f"C:\\Users\\JJD\\Desktop\\Graduate\\dataset\\{TYPE}\\cas\\"

create_shell, platform = False, 'linux'

yh_path = f"/vol7/home/wgx/dataset/{TYPE}/cas/"

linux_path = f"/data/jjd/Graduate/dataset/{TYPE}/cas/"

time_per_cas, core_pre_cas, n_pre_time = 100, 2, 10

print_error = False

cleanup = False

copy_result = True

Airfoil = np.loadtxt(f'{TYPE}Airfoil.csv', delimiter=',', skiprows=1)

Param = np.loadtxt(f'{TYPE}Param.csv', delimiter=',', skiprows=1)

bat = []

shell = []


for j in range(len(Param)):
    mach = float(Param[j, 1])
    aoa = float(Param[j, 2])
    x_com = round(math.cos(math.radians(aoa)), 10)
    y_com = round(math.sin(math.radians(aoa)), 10)
    tag = '{:.0f}_{:.0f}'.format(
        float(mach*10), float(aoa*10))
    f_in = open(jou, 'r')
    lines = f_in.readlines()
    for _ in range(len(lines)):
        lines[_] = re.sub('mach', str(mach), lines[_])
        lines[_] = re.sub('x_com', str(x_com), lines[_])
        lines[_] = re.sub('y_com', str(y_com), lines[_])
        lines[_] = re.sub('template', tag, lines[_])

    for i in range(len(Airfoil)):
        dir_name = int(Airfoil[i, 0])
        if platform == 'linux':
            new_path = f'{linux_path}{dir_name}/'
        elif platform == 'yh':
            new_path = f'{yh_path}{dir_name}/'
        elif platform == 'windows':
            new_path = f'{windows_path}{dir_name}\\'
        if get_jou:
            if not os.path.exists((f'{new_path}J_{tag}.jou')):
                f_out = open(f'{new_path}J_{tag}.jou', 'w')
                f_out.writelines(lines)
                f_out.close()

        if create_bat:
            if not os.path.exists(f'{new_path}{tag}.cas.h5'):
                op = 'cd {new_path}; echo {new_path}:{tag} start; "C:\Program Files\ANSYS Inc\\v211\\fluent\\ntbin\win64\\fluent.exe" 2ddp -t4 -g -mpi=default -i " .\\J_{tag}.jou" &\n'
                bat.append(op)


        if create_shell:
            #!/bin/sh
            if not os.path.exists((f'{new_path}{tag}_res.txt')):
                if platform == 'linux':
                    op = f'cd {new_path}; echo {new_path}:{tag} start; timeout {time_per_cas} /home/jjd/ansys_inc/v211/fluent/bin/fluent -r21.1.0 2ddp -t{core_pre_cas} -g -mpi=openmpi -i ./J_{tag}.jou 1>{linux_path}{dir_name}.log 2>&1  & \n'
                elif platform == 'yh':
                    op = f'cd {new_path}; echo {new_path}:{tag} start; timeout {time_per_cas} /vol7/home/public/software/ansys17/v172/fluent/bin/fluent -r17.2.0 2ddp -t{core_pre_cas} -g -cnf /vol7/home/wgx/dataset/${{SLURM_JOBID}}/${{SLURM_JOBID}}.log -i ./J_{tag}.jou -mpi=intel 1>{yh_path}{dir_name}.log 2>&1  & \n'
                shell.append(op)
                if len(shell) % n_pre_time == 0:
                    shell.append(f'sleep {time_per_cas}\n')
                if len(shell) % (10*n_pre_time) == 1:
                    shell.append('date\n')

        if copy_result:
            if os.path.exists(f'{new_path}{tag}_res.txt') and not os.path.exists(f'{TYPE}/cfd_result/{j}_{dir_name}.txt'):
                copy(f'{new_path}{tag}_res.txt', f'{TYPE}/cfd_result/{j}_{dir_name}.txt')

        if cleanup and j==0:
            os.system(f'rm {linux_path}*.log')
            if os.path.exists(new_path):
                cmd = f'cd {new_path}; rm *.trn *.sh *.log #* '#' *.jou *.h5 *.txt *.out '
                os.system(cmd)


    f_in.close()

if create_bat:
    bat_f = open(fluent_bat, 'w')
    bat_f.writelines(bat)
    bat_f.close()

if create_shell:
    if platform == 'linux':
        shell_f = open(fluent_shell, 'w')
        shell_f.writelines(shell)
        shell_f.close()
    elif platform == 'yh':
        copy(yh_shell, fluent_shell)
        shell_f = open(fluent_shell, 'a+')
        shell_f.writelines(shell)
        shell_f.write('mv ../slurm -${SLURM_JOBID}.out .\ncd ..\n')
        shell_f.close()
