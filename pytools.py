import numpy as np
import os
import pydicom
import matplotlib
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def read_raw_data(file_name, w, h):

    file_temp = np.fromfile(file_name, dtype='float32', sep="")
    slice = int(np.size(file_temp) / w / h)

    if slice == 1:
        file_temp = np.reshape(file_temp, [w, h])

    else:
        file_temp = np.reshape(file_temp, [slice, w, h])

    return slice, file_temp

def read_raw_data_all(dir, w=512, h=512, start_index=8, end_index=-4):

    file_list = os.listdir(dir)
    file_list.sort(key=lambda x: int(x[start_index: end_index]))
    slice = len(file_list)

    file_vol = np.zeros([slice, w, h], dtype=np.float32)

    for index in range(slice):

        file_temp = np.fromfile(dir + file_list[index], dtype='float32', sep="")
        file_temp = file_temp.reshape([w, h])
        file_vol[index, :, :] = file_temp

    return file_vol

def dicomreader(filename):
    info = pydicom.read_file(filename)
    img = np.float32(info.pixel_array)
    return info, img

def listsorter(dir, strat_index, end_index):
    list = os.listdir(dir)
    list.sort(key=lambda x: int(x[strat_index: end_index]))
    return list

def read_dicom_all(file_dir, sort_start, sort_end):

    file_names = listsorter(file_dir, strat_index=sort_start, end_index=sort_end)
    slice_number = len(file_names)
    volume = np.zeros([slice_number, 512, 512], dtype=np.float32)
    for index in range(slice_number):
        _, img = dicomreader(file_dir + file_names[index])
        # img = np.flipud(img)
        # plt.imshow(img, cmap=plt.cm.gray)
        # plt.show()
        volume[index, :, :] = img

    return volume


def make_dirs(dir_path):

    if os.path.exists(dir_path) == False:
        os.makedirs(dir_path)

def addPossionNoisy(CleanProj, I0=1e6):
    
    MaxValueProj = np.max(CleanProj)
    TempProj = CleanProj / MaxValueProj
    # to detector count
    TempProj = I0 * np.exp(-TempProj)
    # add poison noise
    # sinogramCT_C = np.zeros_like(sinogramCT)
    NoiseProj = np.random.poisson(TempProj)
    # for i in range(sinogramCT_C.shape[0]):
    #     for j in range(sinogramCT_C.shape[1]):
    #         sinogramCT_C[i, j] = np.random.poisson(sinogramCT[i, j])
    # for i in range(sinogramCT_C.shape[0]):
    #     sinogramCT_C[i, :] = np.random.poisson(sinogramCT[i, :])
    # to density
    NoiseProj = NoiseProj / I0
    NoiseProj = -MaxValueProj * np.log(NoiseProj)

    return NoiseProj