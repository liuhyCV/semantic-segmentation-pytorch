import sys
import os
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.repo_name = 'semantic-segmentation-pytorch'
""" The absolute path"""
C.abs_dir = os.path.realpath(".")
""" The name of this folder"""
C.config_dir = C.abs_dir.split(os.path.sep)[-1]
""" segm dir """
C.seg_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]
""" The log root folder"""
C.log_dir = C.seg_dir + '/log/' + C.config_dir
""" The log """
C.log_dir_link = os.path.join(C.abs_dir, 'log')
""" The exp_pref for this experiment """
C.exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
""" The snapshots folder"""
C.snapshot_dir = C.log_dir + '/snapshot/'
""" The log file path"""
C.log_file = C.log_dir + '/log_' + C.exp_time + '.log'
C.link_log_file = C.log_dir + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + C.exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

C.img_root_folder = "/unsullied/sharefs/liuhuanyu/workspace/VOC2012_AUG/"
C.ann_root_folder = "/unsullied/sharefs/liuhuanyu/workspace/VOC2012_AUG/"
C.train_source = "/unsullied/sharefs/liuhuanyu/workspace/VOC2012_AUG/config/train.txt"
C.eval_source = "/unsullied/sharefs/liuhuanyu/workspace/VOC2012_AUG/config/val.txt"
C.test_source = "/unsullied/sharefs/liuhuanyu/workspace/VOC2012_AUG/config/voc12_test.txt"


C.pretrain_model = os.path.join(C.seg_dir, 'pretrain_model/resnet101-5d3b4d8f.pth')


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


#add_path(os.path.join(C.seg_dir, 'utils'))
add_path(C.seg_dir)
#add_path(os.path.join(C.seg_dir, 'utils'))


if __name__ == '__main__':
    for i in C:
        print(i, C[i])
    # parser = argparse.ArgumentParser()



