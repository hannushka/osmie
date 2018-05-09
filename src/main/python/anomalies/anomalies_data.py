import os 
import random

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.rsplit('/', 3)[0]
FILENAME = '%s/data/anomaliesData.csv' % (dir_path)
ENC_FILENAME = '%s/data/anomaliesEncodedData.csv' % (dir_path)
TRAIN_FILENAME = '%s/data/anomaliesTrainData.csv' % (dir_path)
TEST_FILENAME = '%s/data/anomaliesTestData.csv' % (dir_path)

def get_speed_class(speed):
    try:
        x = int(speed)
        if x > 0 and x < 30:
            return 0
        if x >= 30 and x < 70:
            return 1
        return 2
    except ValueError:
        if speed == "walk" or speed == "dk:rural":
            return 3
    return 4

def get_surface_class(surface):
    if (not surface):
        return 5
    switch = {
        "paved":0,
        "asphalt":0,
        "concrete":0,
        "concrete:lanes":0,
        "concrete:plates":0,            
        "paving_stones":1,
        "sett":1,
        "unhewn_cobblestone":1,
        "cobblestone":1,
        "metal": 2,
        "wood": 3,
        "unpaved": 4,
        "compacted":4,
        "fine_gravel":4,
        "gravel":4,
        "pebblestone":4,
        "dirt":4,
        "earth":4,
        "grass":4,
        "grass_paver":4,
        "gravel_turf":4,
        "ground":4,
        "mud":4,
        "sand":4,
        "woodchips":4,
        "salt":4,
        "snow":4,
        "ice":4
    }
    if surface in switch:
        return switch[surface]
    return 5

def get_way_class(way):
    if (not way):
        return 7
    switch = {
        "motorway":0,
        "motorway_link":0,
        "trunk":0,
        "trunk_link":0,
        "primary":1,
        "primary_link":1,
        "secondary":1,
        "secondary_link":1,
        "tertiary":1,
        "tertiary_link":1,
        "unclassified":1,            
        "residential":2,
        "pedestrian":3,
        "living_street":3,
        "footway":4,
        "bridleway":4,
        "steps":4,
        "path":4,            
        "cycleway":5
    }
    if way in switch:
        return switch[way]
    return 6

def get_name_class(name):
    suffixes = {
        'vej': 1,
        'gade': 2,
        'sti': 3,
        'boulevard': 4,
        'esplanad': 5,
        'torv': 6,
        'vang': 7,
        'bakk': 8,
        'park': 9,
        'bro': 10,
        'led': 11,
        'væng': 12,
        'plads': 13,
        'høj': 14,
        'passage': 15,
        'sted': 16,
        'bane': 17,
        'træde': 18,
        'promenad': 19,
        'spor': 20,
        'hav': 21,
        'alle': 22,
        'allé': 22
    }
    for suff in suffixes:
        if suff in name:
            return suffixes[suff]
    return 23

if __name__ == '__main__':
    final_data = []
    ways = {}
    #{way-name: {surface:set(surfaces shown), speed:set(speeds shown)...}, way-name2...}
    with open(ENC_FILENAME, 'w', encoding='utf-8') as f2:
        with open(FILENAME, 'r', encoding='utf-8') as f:
            test_data = set()
            for line in f:
                # id,,,name,,,maxspeed,,,surface,,highway
                inp = line.strip().split(',,,')
                if len(inp) < 5:
                    continue
                name = inp[1]
                speed = get_speed_class(inp[2])
                surface = get_surface_class(inp[3])
                way = get_way_class(inp[4])
                new_line = "%s,,,%d,,,%d,,,%d,,,%d" % (inp[0], get_name_class(name), speed, surface, way)
                test_data.add(new_line + ",,,0")
                f2.write(new_line+'\n')
                if name in ways:
                    data = ways[name]
                else:
                    data = {
                        'id':inp[0],
                        'speed':set(),
                        'surface':set(),
                        'way':set()
                    }
                data['speed'].add(speed)
                data['surface'].add(surface)
                data['way'].add(way)
                ways[name] = data
            
            for name in ways.keys():
                data = ways[name]
                pos_speeds = ['']
                for i in range(5):
                    if i not in data['speed']:
                        pos_speeds.append(i)
                pos_surf = ['']
                for i in range(7):
                    if i not in data['surface']:
                        pos_surf.append(i)
                pos_ways = ['']
                for i in range(8):
                    if i not in data['way']:
                        pos_ways.append(i)
                new_data = [data['id']]
                new_data += [get_name_class(name)]
                new_data += [pos_speeds[random.randint(0,len(pos_speeds)-1)]]
                new_data += [pos_surf[random.randint(0,len(pos_surf)-1)]]
                new_data += [pos_ways[random.randint(0,len(pos_ways)-1)]]
                new_data += [1]
                new_data = ',,,'.join(map(str, new_data))
                final_data.append(new_data)
                final_data.append(test_data.pop())
        length = int((3/4)*len(final_data))
        train_data = final_data[:length]
        test_data = final_data[length+1:]
        train_data = '\n'.join(train_data)
        test_data = '\n'.join(test_data)
        with open(TRAIN_FILENAME,'w') as file:
            file.write(train_data)
        with open(TEST_FILENAME,'w') as file:
            file.write(test_data)
