# coding=utf-8

import pickle
import regex as re
from sklearn.metrics import classification_report, confusion_matrix
import os
import argparse

def get_data(name):
    index_list = []
    type_list = []
    s_index = name.find('s')
    while s_index != -1:
        if (s_index+1) < len(name) and name[s_index+1] == 's': # ss
            s_index += 1
            type_list.append('ss')
        else: # s
            type_list.append('s')
        index_list.append(s_index-1)
        s_index = name.find('s', s_index+1)
    return index_list, type_list


def get_chars(char_dict, name, index_list, type_list, j):
    chars = []
    index = index_list[j]
    s_true = 0
    chars = ['\t', '\n']
    if type_list[j] == 's':
        chars[0] = name[index-1]
        chars[1] = name[index+1]
    else:
        s_true = 1
        chars[0] = name[index-2]
        chars[1] = name[index+1]
    ret = []
    for c in chars:
        ret.append(char_dict[c])
    return ret, s_true


def merge_changes(word, changes):
    s_list = re.findall("s+", word)
    word = word.replace("ss", "s")
    wordlist = word.split("s")
    for x in range(0, len(changes)):
        if not changes[x]:
            changes[x] = s_list[x]
    changes.append("")
    return ''.join([a+b for (a, b) in zip(wordlist, changes)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='Name')
    args = parser.parse_args()
    
    name = args.name.lower()

    path = os.path.dirname(os.path.realpath(__file__))

    char_dict = pickle.load( open( path+"/char_dict.p", "rb" ) )
    map_dict = {}
    for it in char_dict.items():
        map_dict[it[1]] = it[0]

    clf = pickle.load( open( path+"/s_classifier.p", "rb" ) )

    test_data = []
    name = ''.join(['\t',name,'\n'])
    index_list, type_list = get_data(name)
    for j in range(0, len(type_list)):
        chars, s_true = get_chars(char_dict, name, index_list, type_list, j)
        test_data.append(chars)
    name = name[1:-1]

    predicts = clf.predict_proba(test_data)

    filtered_predicts = []
    mean_prob = []
    for x in range(0, len(predicts)):
        diff = abs(predicts[x][0]-predicts[x][1])
        if diff > 0:
            if predicts[x][0] < predicts[x][1]: #ss
                filtered_predicts.append('ss')
                mean_prob.append(diff)
            else:
                filtered_predicts.append('s')
                mean_prob.append(diff)
        else:
            filtered_predicts.append(-1)
    if len(mean_prob) > 0:
        print(sum(mean_prob)/len(mean_prob))
    else:
        print(0)

    if len(index_list) > 0:
        for x in range(0, len(filtered_predicts)):
            if filtered_predicts[x] == -1:
                filtered_predicts[x] = type_list[x]
        print(merge_changes(name, filtered_predicts))
    else:
        print(name)

