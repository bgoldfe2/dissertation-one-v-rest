# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023

import pandas as pd
import numpy as np
from Model_Config import traits
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pprint
import json
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score

def test_mv(fldr):
    
    mdl = pd.read_csv(''.join([fldr, 'Age-test_metrics.csv']))
    print(mdl.head())

    test_y = mdl['target']
    pred_y = mdl['y_pred']

    cp = accuracy_score(test_y, pred_y)

    print(cp)


if __name__=="__main__":
    test_run = '../Runs/2023-08-24_17_46_26--roberta-base/Output/'

    test_mv(test_run)

