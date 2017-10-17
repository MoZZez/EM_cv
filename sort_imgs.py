'''
The script sorting given unlabeled imgs in two classes.
1) Crystall ice presist in image
2)No crystall ice just amorphous.
And save them in a given folder in a class folders 'ice' and 'good'.
 
Instruction:

Launch via:
python sort_imgs.py data_path out_path model_path dict_path
See meanings of args in description of 'main' func.
'''
from warnings import warn
from shutil import copy
import os
from os.path import join, exists
from glob import glob
import cv2

import numpy as np
import EMAN2

from ice_filter import models_api as ma

from time import time
import json

import sys

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

#TODO: Do elegant argv handling with getopt and stuf.
def main(argv=None):
    '''
    Supposing argv contains:
    data_path - path to a dir with some .mrc files for classification.
    output_path - path to a dir where will be created two folders: 'good' and 'ice'.
                  So .mrc files of a each class will be saved in the folder with corresponding name.
    model_path - optional argument path to serialized model. Default, will try ./model.pkl(in current dir)
                 
    dict_path - oprional argument path to encoding dictionary. Default, will try to load ./dict.json(current dir).
    '''
    if argv is None:
        argv = sys.argv
        
    argv_len = len(argv)
    
    if argv_len < 3:
        raise ValueError('Set input and ouput paths!')
    
    data_path = argv[1]
    output_path = argv[2]
    
    if argv_len >= 4:
        model_path = argv[3]
    else:
        model_path = 'model.pkl'
    
    if argv_len >=5:
        dict_path = argv[4]
    else:
        dict_path = 'dict.json'
        
    model = joblib.load(model_path)
    
    with open(dict_path,'r') as f:
        encoding_dict = json.load(f)
    #once we loaded dict it`s key may change from ints to str.
    #so we need to reset em to right type.
    encoding_dict = {int(key):encoding_dict[key] for key in encoding_dict.keys()}
    keys = encoding_dict.keys()
    print type(keys[0])
    
    
    if not exists(data_path):
        raise ValueError('Data path does`not exits!')
    
    if not exists(output_path):
        os.makedirs(output_path)
    
    goods_path = join(output_path, 'good')
    ice_path = join(output_path, 'ice')
    
    if not exists(goods_path):
        os.mkdir(goods_path)
    else:
        warn('Class dir "good" already presist in data folder!')
        
    
    if not exists(ice_path):
        os.mkdir(ice_path)
    else:
        warn('Class dir "ice" already presist in data folder!')
        
    #obtaining names from data_dir
    template = join(data_path,'*.mrc')
    names = glob(template)
    print 'Found ', len(names), 'files.'
    
    names_preds_dict = ma.classify_by_names(names,model)
    
    names_labels_dict = ma.decode_preds(names_preds_dict, encoding_dict)
    
    for key in names_labels_dict.keys():
        name = key.split('/')[-1]
        dst_path = join(output_path,names_labels_dict[key],name)
        
        copy(key,dst_path)
        
if __name__ == '__main__':
    sys.exit(main())    
        
    
   
        
    
    
