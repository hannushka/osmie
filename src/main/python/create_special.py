from random import choice, shuffle
import copy
FILE_OLD = 'data/SuperDataUnique.csv'
FILE_NEW = 'data/dataAnomalies.csv'


def parse_super_data():
    data = {'maxspeed':set(), 'surface':set(), 'sidewalk':set(), 'highway':set(), 'oneway':set(), 'count': 0}
    name_data = {}
    possible_tags =copy.deepcopy(data)
    true_data = []
    with open(FILE_OLD, 'r') as f:
        for line in f:
            line = line.strip()
            tags = line.split(',,,')
            name = tags[-1].lower()
            true_data.append(',,,'.join([name] + tags[1:-1] + ['1']))
            if name not in name_data.keys():
                name_data[name] = copy.deepcopy(data)
            update_dict(name_data[name], tags)
            update_dict(possible_tags, tags)

    fake_name_data = []
    for name in name_data.keys():
        fake_data(name_data[name], fake_name_data, possible_tags, name)

    fake_name_data.extend(true_data)
    shuffle(fake_name_data)

    with open(FILE_NEW, 'w') as f:
        f.write('\n'.join(fake_name_data))


def update_dict(data_dict, tags):
    data_dict['maxspeed'].add(tags[1].lower())
    data_dict['surface'].add(tags[2].lower())
    data_dict['highway'].add(tags[3].lower())
    data_dict['sidewalk'].add(tags[4].lower())
    data_dict['oneway'].add(tags[5].lower())
    data_dict['count'] += 1


def fake_data(true_data, fake_dict, possible_tags, name):
    count = true_data['count']
    maxspeed = list(possible_tags['maxspeed'].difference(true_data['maxspeed'])) + ['-1']
    surface = list(possible_tags['surface'].difference(true_data['surface'])) + ['-1']
    highway = list(possible_tags['highway'].difference(true_data['highway'])) + ['-1']
    sidewalk = list(possible_tags['sidewalk'].difference(true_data['sidewalk'])) + ['-1']
    oneway = list(possible_tags['oneway'].difference(true_data['oneway'])) + ['-1']

    for _ in range(count):
        line = ',,,'.join([name, choice(maxspeed), choice(surface), choice(highway),
                           choice(sidewalk), choice(oneway), '0'])
        fake_dict.append(line)






if __name__ == '__main__':
    parse_super_data()