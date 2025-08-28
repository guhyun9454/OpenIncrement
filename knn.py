#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 10:11:36 2022

@author: zhi
"""

import torch
import pickle
import numpy as np
from collections import Counter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from data_loader import iCIFAR100, iCIFAR10, mnist
from dataset import customDataset
from resnet_big import SupConResNet
from mlp import MLP
from itertools import chain

from sklearn import metrics


def normalFeatureReading(data_loader, model):
    features = []
    labels = []

    with torch.no_grad():
        for i, (img, l) in enumerate(data_loader):
            outputs = model(img)
            out_np = outputs.detach().cpu().numpy()
            if isinstance(l, torch.Tensor):
                lab_np = l.detach().cpu().numpy()
            else:
                lab_np = np.array(l)
            for j in range(out_np.shape[0]):
                features.append(out_np[j])
                labels.append(int(lab_np[j]))
        
    return features, labels
        

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]


def KNN(test_feature, exemplars, K):
    
    exemplar_features, exemplar_labels = exemplars
    
    # calculate similarity
    # test_feature = np.tile(test_feature, (len(exemplar_features), 1))
    # similarities = np.squeeze(np.matmul(np.array(exemplar_features), test_feature.T))[:,0]
    similarities = []
    for ef in exemplar_features:
        similarity = np.matmul(np.array(ef), test_feature.T)
        similarities.append(similarity)
    similarities = np.squeeze(np.array(similarities))
    
    ind = np.argsort(similarities)[-K:]
    closest_labels = []
    for i in ind:
        closest_labels.append(exemplar_labels[i])
    
    closest_class = most_frequent(closest_labels)
    
    return closest_class


def OSNN(test_feature, exemplars, K):
    
    exemplar_features, exemplar_labels = exemplars
    
    similarities = []
    for ef in exemplar_features:
        similarity = np.matmul(np.array(ef), test_feature.T)
        similarities.append(similarity)
    similarities = np.squeeze(np.array(similarities))
    
    ind = np.argsort(similarities)[-K:]
    closest_labels = []
    for i in ind:
        closest_labels.append(exemplar_labels[i])
    
    occurence_count = Counter(closest_labels)
    if len(occurence_count) == 1:
        if sum(similarities[ind])/len(ind) > Ts:
            return 10, occurence_count.most_common(2)[0][0], sum(similarities[ind])/len(ind)                         # TODO the return number
        else:
            return 10, 1000, sum(similarities[ind])/len(ind)
    closest_class, second_closest_class = occurence_count.most_common(2)[0][0], occurence_count.most_common(2)[1][0]
    i1 = [i for i, x in enumerate(closest_labels) if closest_labels[i] == closest_class]
    i2 = [i for i, x in enumerate(closest_labels) if closest_labels[i] == second_closest_class]
    closest_ind = ind[i1]
    second_closest_ind = ind[i2]
    closest_similarity = np.sum(similarities[closest_ind])
    second_closest_similarity = np.sum(similarities[second_closest_ind])
    
    R = closest_similarity / second_closest_similarity
    if R < Tr:                                                
        return R, 1000, sum(similarities[ind])/len(ind)
    else:
        return R, closest_class, sum(similarities[ind])/len(ind)
    

def compareLabels(estLabels, trueLabels):
    
    assert len(estLabels) == len(trueLabels)
    unEquals = 0
    for i in range(len(estLabels)):
        if estLabels[i] != trueLabels[i]:
            unEquals += 1
            
    return unEquals



if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10','cifar100','mnist'])
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--Ts', type=float, default=0.85)
    parser.add_argument('--Tr', type=float, default=1.9)
    parser.add_argument('--data_folder', type=str, default='../datasets')
    parser.add_argument('--features_dir', type=str, default='./features')
    parser.add_argument('--fixed_memory', type=int, default=2000)
    parser.add_argument('--memory_per_class', type=int, default=50)
    parser.add_argument('--exemplar_file', type=str, default='', help='path to exemplar file from incremental step')
    parser.add_argument('--alfa', type=float, default=0.2)
    parser.add_argument('--temp', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()

    Ts = args.Ts
    Tr = args.Tr
    K = args.K
    num_classes = args.num_classes
    dataset = args.dataset
    classes = [i for i in range(num_classes)] + [i for i in range(90, 100)]
    exemplar_feature_path = os.path.join(
        args.features_dir,
        'exemplar_{}_class_{}_{}_memorysize_{}_alfa_{}_temp_{}_mem_{}'.format(dataset, num_classes, args.model, args.memory_per_class, args.alfa, args.temp, args.fixed_memory)
    )
    
    model = SupConResNet(args.model) if args.model != 'mlp' else MLP()
    ckpt_name = '{}_{}_class_{}_{}_lr_{}_epochs_{}_bsz_{}_temp_{}_alfa_{}_mem_{}_incremental/last.pth'.format(
        'SupCon', dataset, num_classes, args.model, args.learning_rate, args.epochs, args.batch_size, args.temp, args.alfa, args.fixed_memory)
    ckpt_path = os.path.join('./save/SupCon/{}_models'.format(dataset), ckpt_name)
    if not os.path.isfile(ckpt_path):
        # fallback to legacy naming
        legacy = '{}_{}_class_{}_{}_lr_{}_epoch_{}_bsz_{}_temp_{}_incremental/last.pth'.format(
            'SupCon', dataset, num_classes, args.model, args.learning_rate, args.epochs, args.batch_size, args.temp)
        legacy_path = os.path.join('./save/SupCon/{}_models'.format(dataset), legacy)
        ckpt_path = legacy_path if os.path.isfile(legacy_path) else ckpt_path
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v

    state_dict = new_state_dict
    model = model.cpu()
    model.load_state_dict(state_dict)
    model.eval()
    
    exemplar_features = None
    exemplar_labels = None
    # 1) preferred: provided exemplar_file (raw images/labels) -> compute features now
    if args.exemplar_file and os.path.isfile(args.exemplar_file):
        with open(args.exemplar_file, 'rb') as f:
            exemplar_sets, exemplar_labels, _, _ = pickle.load(f)
        # build transform consistent with dataset
        if dataset == 'cifar100' or dataset == 'cifar10':
            normalize = transforms.Normalize((0.5071, 0.4867, 0.4408) if dataset=='cifar100' else (0.4914,0.4822,0.4465),
                                             (0.2675, 0.2565, 0.2761) if dataset=='cifar100' else (0.2023,0.1994,0.2010))
            tfm = transforms.Compose([transforms.ToTensor(), normalize])
        else:
            if args.model == 'mlp':
                normalize = transforms.Normalize((0.1307,), (0.3081,))
                tfm = transforms.Compose([transforms.ToTensor(), normalize])
            else:
                normalize = transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081))
                tfm = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), normalize])
        ex_dataset = customDataset(exemplar_sets, exemplar_labels, transform=tfm)
        ex_loader = torch.utils.data.DataLoader(ex_dataset, batch_size=64, shuffle=False, num_workers=2)
        feats, labs = normalFeatureReading(ex_loader, model)
        exemplar_features, exemplar_labels = feats, labs
    # 2) fallback: precomputed features file if exists
    elif os.path.isfile(exemplar_feature_path):
        with open(exemplar_feature_path, "rb") as f:
            exemplar_features, exemplar_labels = pickle.load(f)
    # 3) last resort: build random exemplars from training data on-the-fly
    else:
        if dataset == 'cifar100':
            base_ds = iCIFAR100(root=args.data_folder, train=True, classes=range(num_classes), download=True, transform=None)
            data, labels = base_ds.train_data, base_ds.train_labels
        elif dataset == 'cifar10':
            base_ds = iCIFAR10(root=args.data_folder, train=True, classes=range(num_classes), download=True, transform=None)
            data, labels = base_ds.train_data, base_ds.train_labels
        else:
            base_ds = mnist(root=args.data_folder, train=True, classes=range(num_classes), download=True, transform=None)
            data, labels = base_ds.traindata, base_ds.trainlabels
        import numpy as np
        exemplar_sets = []
        exemplar_labels = []
        for c in range(num_classes):
            idxs = np.where(np.array(labels) == c)[0][:args.memory_per_class]
            if len(idxs) == 0:
                continue
            for i in idxs:
                exemplar_sets.append(np.array(data[i]))
                exemplar_labels.append(c)
        exemplar_sets = np.array(exemplar_sets)
        # transform same as above
        if dataset == 'cifar100' or dataset == 'cifar10':
            normalize = transforms.Normalize((0.5071, 0.4867, 0.4408) if dataset=='cifar100' else (0.4914,0.4822,0.4465),
                                             (0.2675, 0.2565, 0.2761) if dataset=='cifar100' else (0.2023,0.1994,0.2010))
            tfm = transforms.Compose([transforms.ToTensor(), normalize])
        else:
            if args.model == 'mlp':
                normalize = transforms.Normalize((0.1307,), (0.3081,))
                tfm = transforms.Compose([transforms.ToTensor(), normalize])
            else:
                normalize = transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081))
                tfm = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), normalize])
        ex_dataset = customDataset(exemplar_sets, exemplar_labels, transform=tfm)
        ex_loader = torch.utils.data.DataLoader(ex_dataset, batch_size=64, shuffle=False, num_workers=2)
        feats, labs = normalFeatureReading(ex_loader, model)
        exemplar_features, exemplar_labels = feats, labs
        

    if dataset == "cifar100":
        transform = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                             (0.2675, 0.2565, 0.2761)),])
        test_set = iCIFAR100(root=args.data_folder, train=False,                        
                             classes=classes,
                             download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                             shuffle=True, num_workers=2)
    elif dataset == "cifar10":
        transform = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                             (0.2023, 0.1994, 0.2010)),])
        test_set = iCIFAR10(root=args.data_folder, train=False,                        
                            classes=classes,
                            download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                             shuffle=True, num_workers=2)
    elif dataset == "mnist":
        transform = transforms.Compose([#transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,),
                                                             (0.3081,)),])
        test_set = mnist(root=args.data_folder, train=False,
                         classes=classes, download=True,                                #####
                         transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                             shuffle=True, num_workers=2)
    
    test_features, test_labels = normalFeatureReading(test_loader, model)
    
    predictions = []
    RsIn = []
    RsOut = []
    SIn = []
    SOut = []
    for test_feature, test_label in zip(test_features, test_labels):
        
        #closest_class = KNN(test_feature, (exemplar_features, exemplar_labels), K)
        R, closest_class, similarities= OSNN(test_feature, (exemplar_features, exemplar_labels), K)
        if test_label in range(num_classes):
            RsIn.append(R)
            SIn.append(similarities)
        else:
            RsOut.append(R)
            SOut.append(similarities)
        predictions.append(closest_class)
    
    # OOD metric: AUROC & FPR@TPR95 using average similarity as ID score
    id_scores = np.array(SIn)
    ood_scores = np.array(SOut)
    binary_labels = np.concatenate([np.ones(id_scores.shape[0]), np.zeros(ood_scores.shape[0])])
    all_scores = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = metrics.roc_curve(binary_labels, all_scores, drop_intermediate=False)
    auroc = metrics.auc(fpr, tpr)
    idx_tpr95 = np.abs(tpr - 0.95).argmin()
    fpr_at_tpr95 = fpr[idx_tpr95]
    print(f"[OSNN]: AUROC {auroc * 100:.2f}% | FPR@TPR95 {fpr_at_tpr95 * 100:.2f}%")