#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 20:55:55 2022

@author: zhi
"""

import pickle
import numpy as np
from PIL import Image

from torch.autograd import Variable 
import random


def read_features(images, model, transform):
    
    model = model.eval()
    #model.cuda()
    features = []
    f = []
    for img in images:
        # 입력 배열의 채널 수에 맞게 PIL 이미지 생성 (자동 모드)
        x = Variable(transform(Image.fromarray(np.squeeze(img))), volatile=True)
        x = x.cuda()
        feature = model(x.unsqueeze(0))
        feature = feature.cpu().data.numpy()
        f.append(feature)
        feature = feature / np.linalg.norm(feature) # Normalize
        features.append(feature[0])
        
    features = np.array(features)
    return features
    


def classExemplars_euclidean(m, images, model, transform):
 
    features = read_features(images, model, transform)
    center = centerComputing(features)
    centers = np.tile(center, (len(features), 1))
    distances = np.linalg.norm((features-centers), axis=1)

    ind = np.argsort(distances)[:m]
    exemplar_set = np.array(images)[ind]
    exemplar_features = features[ind]

    return exemplar_set, exemplar_features, center
    

def classExemplars_similar(m, images, model, transform):
    
    features = read_features(images, model, transform)
    center = centerComputing(features)
    centers = np.tile(center, (len(features), 1))
    similarities = np.matmul(features, centers.T)[:, 0]
    
    ind = np.argsort(np.abs(similarities))[:m]
    exemplar_set = np.array(images)[ind]
    exemplar_features = features[ind]

    return exemplar_set, exemplar_features, center


def classExemplars_random(m, images, model=None, transform=None):
    
    #features = read_features(images, model, transform)
    #center = centerComputing(features)
    
    ind = random.sample(range(len(images)), m)
    exemplar_set = np.array(images)[ind]
    #exemplar_features = features[ind]
    
    return exemplar_set


def createExemplars(opt, original_dataset, model_old=None, transform=None):
    
    exemplar_sets = []
    exemplar_labels = [] 
    exemplar_features_sets = []
    exemplar_centers = []
    # 현재 엑셈플러를 만들 클래스 수(증분 스텝에서 전체(0..num_classes-1)로 갱신 가능)
    num_classes_for_exemplar = getattr(opt, 'exemplar_num_classes', opt.num_init_classes)
    if opt.fixed_memory == 0:
        opt.memory_per_class = opt.memory_size
    else:
        opt.memory_per_class = max(1, opt.fixed_memory // num_classes_for_exemplar)
        
    for c in range(0, num_classes_for_exemplar):
        print("Class: ", c)
        c_dataset = original_dataset.get_image_class(c)
        # 유클리드 중심 기반 샘플링 사용 (Isometric에 가까운 중앙 근접 선택)
        exemplar_set, _, _ = classExemplars_euclidean(int(opt.memory_per_class), c_dataset, model_old, transform)
        #exemplar_center = centerComputing(exemplar_features)
        #exemplar_centers.append(exemplar_center)
        exemplar_sets.append(exemplar_set)
        #exemplar_features_sets.append(exemplar_features)
        exemplar_labels = exemplar_labels + [c]*int(opt.memory_per_class)
        
    total_exemplars = opt.memory_per_class * num_classes_for_exemplar
    exemplar_sets = np.reshape(np.array(exemplar_sets), (total_exemplars, opt.img_size, opt.img_size, 3)) 
    exemplar_labels = np.squeeze(np.array(exemplar_labels))   
    
    with open(opt.exemplar_file, "wb") as f:
        pickle.dump((exemplar_sets, exemplar_labels, exemplar_features_sets, exemplar_centers), f)
    
    return exemplar_sets, exemplar_labels, exemplar_centers


def centerComputing(features):
    
    return np.mean(features, 0)


