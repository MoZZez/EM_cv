'''
Instruction:
Launch via:
python train_model.py data_path model_path dict_path

data_path - path to a folder with two subfolders 'good' and 'ice'.
            They contain mrc files which will be used for train.

model_path - path like some_path/model.pkl to a file in which model will be saved.
             Optional argument, default is current directory ./model.pkl.

dict_path - path via some_path/dict.json
            There will be saved dictionary mathcing labels and labels names.
            Oprional argument, default is current directory ./dict.json


'''
from os.path import join
from glob import glob
import cv2

import numpy as np
import EMAN2

from ice_filter import models_api as ma

from time import time
import json

import sys

from sklearn.linear_model import LogisticRegression

#TODO: Do elegant argv handling with getopt and stuf.
def main(argv=None):
    '''
    Supposing argv contains:
    1) data_path
    2) model_path(optional)
    3) dict_path(optional)
    '''
    if argv is None:
        argv = sys.argv
        
    argv_len = len(argv)
    if argv_len == 1:
        raise ValueError('Set data path!')
    #TODO: Replace dirty argv handling.
    data_path = argv[1]
    if argv_len > 2:
        model_path = argv[2]
    else:
        model_path = 'model.pkl'
        
    if argv_len > 3:
        dict_path = argv[3]
    else:
        dict_path = 'dict.json'
    
    #obtaining names from data_dir
    template = join(data_path,'*','*.mrc')
    names = glob(template)   
    
    dataset,labels,encoding_dict = ma.do_dataset(names)
    
    #saving dictionary
    with open(dict_path,'w') as f:
        json.dump(encoding_dict, f)
    
    model = LogisticRegression()
    model,acc = ma.train_new_model(dataset,labels,model,model_path) 
    
    
if __name__ == '__main__':
    sys.exit(main())