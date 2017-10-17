from os.path import join
from glob import glob
import cv2

import numpy as np
from numpy.random import shuffle
import EMAN2

import filtering_tools as ft

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

def do_dataset(names):
    '''
    Supposing we have marked data laying in folder data,
    in two subfolders 'good' and 'ice' and only this way.
    For now way of keeping data is hardcoded.
    #TODO:Make set creator more flexible.
    
    So we got list of full paths to mrc files lying in both direcories.
    Func assembles dataset,labels encoding dictionary
    matching classifier labels to words 'good' and 'ice' and
    returning it.
    
    Args:
         names - list of full paths to mrc files lying in a directories
                 as described above.
    
    returns: dataset(list of feature vectors)
             labels(list of labels)
             encoding_dict dictionary via label:label_name.
    '''
    labels = [name.split('/')[-2] for name in names]
    labels = np.array(labels)

    ice_mask = labels == 'ice'
    good_mask = labels == 'good'
    
    names = np.array(names)
    ice_names = names[ice_mask]
    good_names = names[good_mask]
    
    good_labels = [0]*len(good_names)
    ice_labels = [1]*len(good_names)
    
    encoding_dict = {0 : 'good', 1 : 'ice'}
    
    goods = []
    ices = []
    
    for name in good_names:
        img = EMAN2.EMData(name, 0)
        img_np = EMAN2.EMNumPy.em2numpy(img)
        sample = ft.preprocess_img(img_np)
        goods.append(sample)
        
    for name in ice_names:
        img = EMAN2.EMData(name, 0)
        img_np = EMAN2.EMNumPy.em2numpy(img)
        sample = ft.preprocess_img(img_np)
        ices.append(sample)
        
    dataset = goods + ices
    labels = good_labels + ice_labels
    
    return dataset,labels,encoding_dict

def train_new_model(data, labels, model, model_path, split=True):
    '''
    Once we got marked data,labels and sklearn model this fucntion
    puts all toghether in a training pipeline.
    
    Args:
         data - list of samples.(feature vectors)
         labels - list of numeric(1,2,3) labels
         model - sklearn model instance.
         model_path - str path and name via name.pkl 
                      to save model once it trained.
         split - bool, whether to split data on train ans test sets and
         test trained model on test. Default is True.
    '''
    if split == True:
        X_train,X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=13)
    else:
        X_train = list(data)
        y_train = list(labels)
        
        order = [i for i in range(len(data))]
        shuffle(order)
        
        data = np.array(data)[order].tolist()
        labels = np.array(labels)[order].tolist()
        
    model.fit(X_train,y_train)
    
    if split == True:
        preds = model.predict(X_test)
        acc = accuracy_score(y_test,preds)
    else:
        preds = model.predict(X_train)
        acc = accuracy_score(y_train,preds)
    print 'Models accuracy: ',acc
    
    joblib.dump(model,model_path)
    return model,acc


def classify_by_names(names,model):
    '''
    With a given sklearn model and list of paths to mrc files
    predicts label for each name
    
    Args:
         names - list of paths to mrc files
         model - sklearn model instance.
                 (should predict classes not probabilities)
                 
    returns - dict like name:label
    '''
    samples = []
    for name in names:
        img = EMAN2.EMData(name, 0)
        img_np = EMAN2.EMNumPy.em2numpy(img)
        sample = ft.preprocess_img(img_np)
        samples.append(sample)
        
    preds = model.predict(samples)
    
    names_preds_dict = {name : label for name,label in zip(names,preds)}
    
    return names_preds_dict


def decode_preds(names_preds,encoding_dict):
    '''
    With a given encoding dictionary name:label and
    encoding_dict label:label_name.
    
    Transform name:label to name:label_name.
    
    Args:
         names_preds - dictionary via name:label
         encoding_dict - dictionary via label:label_name.
         
    returns: dictionary via name:label_name.
    '''
    keys =  encoding_dict.keys()
    values = np.unique(names_preds.values())
    names_classes = {}
    for key in names_preds:
        sample_class = encoding_dict[names_preds[key]]
        names_classes[key] = sample_class
        
    return names_classes