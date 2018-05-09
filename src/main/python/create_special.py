from random import choice, shuffle
import copy
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.rsplit('/', 3)[0]
FILE_OLD = '%s/data/anomaliesData.csv' % (dir_path)
FILE_TRAIN =  '%s/data/anomaliesSpecialDataTrain.csv' % (dir_path)
FILE_TEST =  '%s/data/anomaliesSpecialDataTest.csv' % (dir_path)

def parse_super_data():
    data = {'maxspeed':set(), 'surface':set(), 'highway':set(),  'count': 0}
    name_data = {}
    possible_tags = copy.deepcopy(data)
    true_data = []
    with open(FILE_OLD, 'r') as f:
        for line in f:
            line = line.strip()
            tags = line.split(',,,')
            name = tags[1].lower()
            true_data.append(',,,'.join([name] + tags[2:] + ['0']))
            if name not in name_data.keys():
                name_data[name] = copy.deepcopy(data)
            update_dict(name_data[name], tags)
            update_dict(possible_tags, tags)

    fake_name_data = []
    for name in name_data.keys():
        fake_data(name_data[name], fake_name_data, possible_tags, name)

    fake_name_data.extend(true_data)
    shuffle(fake_name_data)

    length = int(len(fake_name_data)*(3/4))
    train_data = fake_name_data[:length]
    test_data = fake_name_data[length:]
    with open(FILE_TRAIN, 'w') as f:
        f.write('\n'.join(train_data))
    with open(FILE_TEST, 'w') as f:
        f.write('\n'.join(test_data))


def update_dict(data_dict, tags):
    data_dict['maxspeed'].add(tags[2].lower())
    data_dict['surface'].add(tags[3].lower())
    data_dict['highway'].add(tags[4].lower())
    data_dict['count'] += 1


def fake_data(true_data, fake_dict, possible_tags, name):
    count = true_data['count']
    maxspeed = list(possible_tags['maxspeed'].difference(true_data['maxspeed'])) + ['']
    surface = list(possible_tags['surface'].difference(true_data['surface'])) + ['']
    highway = list(possible_tags['highway'].difference(true_data['highway'])) + ['']

    for _ in range(count):
        line = ',,,'.join([name, choice(maxspeed), choice(surface), choice(highway),'1'])
        fake_dict.append(line)

if __name__ == '__main__':
    parse_super_data()