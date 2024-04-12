import os
import shutil
import yaml
import json
import pickle

import numpy as np
import SimpleITK as sitk

from typing import List
from multiprocessing import Pool


def open_yaml(file_path, mode="r"):
    with open(file_path, mode) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def save_yaml(data, save_path, mode="w"):
    with open(save_path, mode) as f:
        yaml.dump(data=data, stream=f, allow_unicode=True)


def open_json(file_path, mode="r"):
    with open(file_path, mode) as f:
        data = json.load(f)
    return data


def save_json(data, save_path, mode="w", sort_keys=True, indent=4):
    with open(save_path, mode) as f:
        json.dump(data, f, sort_keys=sort_keys, indent=indent)


def open_pickle(file_path, mode="rb"):
    with open(file_path, mode) as f:
        data = pickle.load(f)
    return data


def save_pickle(data, save_path, mode="wb"):
    with open(save_path, mode) as f:
        pickle.dump(data, f)


def save_itk(data, property, save_path):
    '''
    data is a np.ndarray: trans it to sitk obj and save.
    data is a sitk obj: we think the obj already got its properties, so just save.
    '''
    # print(save_path)
    if isinstance(data, np.ndarray):
        data_obj = sitk.GetImageFromArray(data)
        data_obj.SetDirection(property["direction"])
        data_obj.SetOrigin(property["origin"])
        data_obj.SetSpacing(property["spacing"])
        sitk.WriteImage(data_obj, save_path)
    elif isinstance(data, sitk.Image):
        sitk.WriteImage(data_obj, save_path)
    else:
        raise Exception("Unsupported data types: {}".format(type(data)))


def copy_file_to_dstFolder(srcfile, dstfolder):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        # get the file's name
        _, fname = os.path.split(srcfile)
        if not os.path.exists(dstfolder):
            os.makedirs(dstfolder)
        dst_path = os.path.join(dstfolder, fname)
        # copy file from src path to dst path
        shutil.copy(srcfile, dst_path)
        print("Copy %s -> %s" % (srcfile, dst_path))


def check_workers_alive_and_busy(export_pool: Pool, worker_list: List, results_list: List, allowed_num_queued: int = 0):
    """

    returns True if the number of results that are not ready is greater than the number of available workers + allowed_num_queued
    """
    alive = [i.is_alive() for i in worker_list]
    if not all(alive):
        raise RuntimeError('Some background workers are no longer alive')

    not_ready = [not i.ready() for i in results_list]
    if sum(not_ready) >= (len(export_pool._pool) + allowed_num_queued):
        return True
    return False
