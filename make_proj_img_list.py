import os
import matplotlib.pyplot as plt
import numpy as np
from pytools import *
# import pylab
import time
# from scipy.ndimage import zoom
import os

train_cases = ['L096', 'L143', 'L333', 'L291', 'L286', 'L310', 'L506', 'L109']
valid_cases = ['L192']

views = 72
train_npz_save_dir = '/home/10T/DreamNet/Data/NPZ/TrainProjImgs1e6V' + str(views) + '/'
make_dirs(train_npz_save_dir)

cnt = 0

for case in train_cases:

    startTime = time.time()

    hdct_path = '/home/10T/DreamNet/Data/AAPM/' + case + '/full_1mm/'
    hdct_vol = read_dicom_all(hdct_path, 20, 24)
    hdct_vol = hdct_vol * 0.02 / 1024

    hdproj_vol = read_raw_data_all('/home/10T/DreamNet/Data/sAAPMProj/' + case + '/clean_proj/', w=576, h=736, start_index=0, end_index=-15)
    ldproj_vol = read_raw_data_all('/home/10T/DreamNet/Data/sAAPMProj/' + case + '/noisy_proj_1e6/', w=576, h=736, start_index=0, end_index=-15)
    ldct_vol = read_raw_data_all('/home/10T/DreamNet/Data/sAAPMImg/' + case + '/sparse_ct_v' + str(views) + '_1e6/', w=512, h=512, start_index=0, end_index=-14)

    # plt.imshow(ldproj_vol[150, :, :], cmap=plt.cm.gray)
    # plt.show()

    for slice in range(0, np.size(hdct_vol, 0)):

        hdct_slices = hdct_vol[slice, :, :]
        ldct_slices = ldct_vol[slice, :, :]
        hdproj_slices = hdproj_vol[slice, :, :]
        ldproj_slices = ldproj_vol[slice, :, :]

        # print(np.shape(hdct_slices))
        # print(np.shape(hdproj_slices))
        # print(np.shape(ldproj_slices))

        np.savez(train_npz_save_dir + case + '_slice' + str(slice), ldproj=ldproj_slices, hdproj=hdproj_slices, hdct=hdct_slices, ldct=ldct_slices)
        with open('./train_clear_list_s1e6_v' + str(views) + '.txt', 'a') as f:
            f.write(train_npz_save_dir + case + '_slice' + str(slice) + '.npz\n')
        f.close()

        cnt += 1

    endTime = time.time()

    print('Patient {0} finished, totally got {1} samples, cost {2} seconds.'.format(case, cnt, int(endTime-startTime)))

valid_npz_save_dir = '/home/10T/DreamNet/Data/NPZ/ValidProjImgs1e6V' + str(views) + '/'
make_dirs(valid_npz_save_dir)

cnt = 0

for case in valid_cases:

    startTime = time.time()

    hdct_path = '/home/10T/DreamNet/Data/AAPM/' + case + '/full_1mm/'
    hdct_vol = read_dicom_all(hdct_path, 20, 24)
    hdct_vol = hdct_vol * 0.02 / 1024

    hdproj_vol = read_raw_data_all('/home/10T/DreamNet/Data/sAAPMProj/' + case + '/clean_proj/', w=576, h=736, start_index=0, end_index=-15)
    ldproj_vol = read_raw_data_all('/home/10T/DreamNet/Data/sAAPMProj/' + case + '/noisy_proj_1e6/', w=576, h=736, start_index=0, end_index=-15)
    ldct_vol = read_raw_data_all('/home/10T/DreamNet/Data/sAAPMImg/' + case + '/sparse_ct_v' + str(views) + '_1e6/', w=512, h=512, start_index=0, end_index=-14)

    for slice in range(0, np.size(hdct_vol, 0)):

        hdct_slices = hdct_vol[slice, :, :]
        ldct_slices = ldct_vol[slice, :, :]
        hdproj_slices = hdproj_vol[slice, :, :]
        ldproj_slices = ldproj_vol[slice, :, :]

        np.savez(valid_npz_save_dir + case + '_slice' + str(slice), ldproj=ldproj_slices, hdproj=hdproj_slices, hdct=hdct_slices, ldct=ldct_slices)
        with open('./valid_clear_list_s1e6_v' + str(views) + '.txt', 'a') as f:
            f.write(valid_npz_save_dir + case + '_slice' + str(slice) + '.npz\n')
        f.close()

        cnt += 1

    endTime = time.time()

    print('Patient {0} finished, totally got {1} samples, cost {2} seconds.'.format(case, cnt, int(endTime-startTime)))