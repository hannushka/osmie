import xgboost as xgb
from sklearn.externals import joblib
import regex as re
import pickle
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import sys
import os
from anomalies_data import *
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.rsplit('/', 4)[0]
FILENAME = '%s/data/anomaliesSpecialData.csv' % (dir_path)
OSM_FILENAME = '%s/data/anomaliesData.csv' % (dir_path)
TRAIN_FILE = '%s/data/anomaliesSpecialDataTrain.txt' % (dir_path)
TEST_FILE = '%s/data/anomaliesSpecialDataTest.txt' % (dir_path)
OSM_FILE = '%s/data/anomaliesSpecialDataOSM.txt' % (dir_path)

def generate_train_files():
    with open(FILENAME, 'r') as f:
        lines = []
        for line in f:    
            #name,,,maxspeed,,,surface,,highway,,,0/1
            inp = line.strip().split(',,,')
            suff = get_name_class(inp[0])
            speed = get_speed_class(inp[1])
            surface = get_surface_class(inp[2])
            way = get_way_class(inp[3])
            lines.append('{} 1:{} 2:{} 3:{} 4:{}\n'.format(inp[4],suff,speed,surface,way))
        ind = int(len(lines)*3/4)
        train_data = lines[:ind]
        with open(TRAIN_FILE, 'w') as f1:
            for t in train_data:
                f1.write(t)
        test_data = lines[ind:] 
        test_y = []       
        with open(TEST_FILE, 'w') as f2:
            for t in test_data:
                f2.write(t)
                test_y.append(int(t[0]))
        return test_y

def generate_osm_data():
    ids = []
    with open(OSM_FILENAME, 'r') as f1:
        with open(OSM_FILE, 'w') as f2:
            for line in f1:    
                #id,,,name,,,maxspeed,,,surface,,highway
                inp = line.strip().split(',,,')
                suff = get_name_class(inp[1])
                speed = get_speed_class(inp[2])
                surface = get_surface_class(inp[3])
                way = get_way_class(inp[3])
                ids.append(inp[0])
                f2.write('1:{} 2:{} 3:{} 4:{}\n'.format(suff,speed,surface,way))
    return ids

if __name__ == '__main__':
    test_y = generate_train_files()
    dtrain = xgb.DMatrix(TRAIN_FILE)
    dtest = xgb.DMatrix(TEST_FILE)
    param = {'max_depth': 10, 'silent': 1, 'objective': 'multi:softprob', 'num_class': 2}
    param['nthread'] = 1
    param['eval_metric'] = 'auc'
    num_round = 50
    bst = xgb.train(param, dtrain, num_round)
    
    ids = generate_osm_data()
    osmtest = xgb.DMatrix(OSM_FILE)
    ypred = bst.predict(osmtest)
    anomalies = []
    for i in range(len(ypred)):
        if abs(ypred[i][0]-ypred[i][1]) > 0.9:
            if ypred[i][0] < ypred[i][1]: #1 anomaly
                anomalies.append(ids[i])
    print(len(anomalies))

    preds = bst.predict(dtest)
    filtered_test = []
    filtered_preds = []
    for i in range(0, len(preds)):
        if abs(preds[i][0]-preds[i][1]) > 0:
            if preds[i][0] < preds[i][1]: #ss
                filtered_preds.append(1)
                filtered_test.append(test_y[i])
            else:
                filtered_preds.append(0)
                filtered_test.append(test_y[i])

    print(confusion_matrix(filtered_test, filtered_preds))
    print(classification_report(filtered_test, filtered_preds))
    print(f1_score(filtered_test, filtered_preds, average='macro'))
 