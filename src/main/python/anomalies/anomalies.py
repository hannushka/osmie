from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import regex as re
from sklearn.tree import export_graphviz
import os
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.rsplit('/', 3)[0]
TRAIN_FILENAME = '%s/data/anomaliesTrainData.csv' % (dir_path)
TEST_FILENAME = '%s/data/anomaliesTestData.csv' % (dir_path)
ENC_FILENAME = '%s/data/anomaliesEncodedData.csv' % (dir_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomalies')
    parser.add_argument('anomalies', metavar='type', type=int, nargs='+',
                    help='type. 0: train, 1:test, 2:test on production data')
    args = parser.parse_args()
    TYPE = args.anomalies[0]
    if not TYPE:
        with open(TRAIN_FILENAME, 'r', encoding='utf-8') as f:
            train_data = []
            train_y = []
            for line in f:
                # id,,,name,,,maxspeed,,,surface,,highway,,,0/1
                inp = line.strip().split(',,,')
                if len(inp) < 6:
                    continue
                name = 0
                if inp[1]:
                    name = inp[1]
                speed = 0
                if inp[2]:
                    speed = inp[2]
                surface = 0
                if inp[3]:
                    surface = inp[3]
                way = 0
                if inp[4]:
                    way = inp[4]
                train_data.append([name, speed, surface, way])
                train_y.append(inp[5].strip())
            clf = RandomForestClassifier(n_estimators=20,max_depth=10,n_jobs=-1)
            clf.fit(train_data, train_y)
            pickle.dump( clf, open( "anomalies_clf.p", "wb" ) )
            tree.export_graphviz(clf.estimators_[0], out_file='tree.dot') 
    elif TYPE == 1:
        with open(TEST_FILENAME, 'r', encoding='utf-8') as f:
            test_data = []
            test_y = []
            test_ids = []
            for line in f:
                # id,,,name,,,maxspeed,,,surface,,highway,,,0/1
                inp = line.strip().split(',,,')
                if len(inp) < 6:
                    continue
                test_ids.append(inp[0])
                name = 0
                if inp[1]:
                    name = inp[1]
                speed = 0
                if inp[2]:
                    speed = inp[2]
                surface = 0
                if inp[3]:
                    surface = inp[3]
                way = 0
                if inp[4]:
                    way = inp[4]
                test_data.append([name, speed, surface, way])
                test_y.append(inp[5].strip())
            clf = pickle.load( open( "anomalies_clf.p", "rb" ) )
            predicts = clf.predict(test_data)
            print(predicts)
            print(confusion_matrix(test_y, predicts))
            print(classification_report(test_y, predicts))
    else:
        with open(ENC_FILENAME, 'r', encoding='utf-8') as f:
            test_data = []
            test_ids = []
            for line in f:
                # id,,,name,,,maxspeed,,,surface,,highway,,,0/1
                inp = line.strip().split(',,,')
                if len(inp) < 5:
                    continue
                test_ids.append(inp[0])
                test_data.append([inp[1], inp[2], inp[3], inp[4]])
            clf = pickle.load( open( "anomalies_clf.p", "rb" ) )
            predicts = clf.predict(test_data)
            anomalies = []
            for i in range(len(predicts)):
                if predicts[i] == 1: #anomaly
                    anomalies.append([test_ids[i]] + test_data[i])
            print(len(anomalies), len(test_data))