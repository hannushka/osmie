from numpy.random import choice as random_choice, randint as random_randint, rand, shuffle
import string
import regex as re
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.rsplit('/', 3)[0]
NAME_FILE = '%s/data/SuperDataUnique.csv' % dir_path
AUTO_NOISE_FILE = '%s/data/SuperDataUnique.csv.noised' % dir_path
MANUAL_NOISE_FILE = '%s/data/manualNameData.csv' % dir_path
KORPUS_FILE = '%s/data/korpus_freq_dict.txt.mini' % dir_path
KORPUS_NOISED_FILE = '%s/data/korpus_freq_dict.txt.mini.noised' % dir_path
ALPHABET = '%s/data/dk_alphabet.txt' % dir_path
AMOUNT_NOISE = 0.1
keyboard_lib = {'a': ['s'], 's': ['a', 'd', 'w', 'x'], }
TRANSPOSE = 'T'
REPLACE = 'R'
INSERT = 'I'
DELETE = 'D'


def edits1(word, letters):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [(L + R[1:], 'delete,' + word[len(L)]) for L, R in splits if R]
    transposes = [(L + R[1] + R[0] + R[2:], 'transpose,' + R[0] + "," + R[1]) for L, R in splits if len(R) > 1 and R[0] != R[1]]
    replaces = [(L + c + R[1:], 'replace,' + word[len(L)] + "," + c) for L, R in splits if R for c in letters if c != R[0]]
    inserts = [(L + c + R, 'insert,' + L[-1] + c + R[0]) for L, R in splits if len(L) > 1 and len(R) > 0 for c in letters]
    return set(deletes + replaces + inserts + transposes)


def get_all_changes(list_of_lines, alphabet):
    changes = {DELETE: [], INSERT: [], TRANSPOSE: [], REPLACE: []}
    for line in list_of_lines:
        line = line.lower().strip()
        line = line.split(',,,')
        s1 = line[0]
        s2 = line[1]
        edits_1 = edits1(s1, alphabet)
        res = [(a, b) for (a, b) in edits_1 if a == s2]
        if res:
            res = res[0][1]
            sp2 = res.split(',')  # edited_word,edit_message

            if sp2[0] == 'delete':
                changes[DELETE].append(sp2[1])  # sp2 = [delete,letter]
            elif sp2[0] == 'insert':
                changes[INSERT].append(sp2[1])  # sp2 = [insert,letter]
            elif sp2[0] == 'replace':
                key = res.split(',', 1)[1]  # replace,letter1,letter2
                changes[REPLACE].append(key)
            elif sp2[0] == 'transpose':  # transpose,letter1,letter2
                key = res.split(',', 1)[1]
                changes[TRANSPOSE].append(key)

    return changes


def manual_based_error(change_list, selection_list, word):
    key = random_choice(selection_list)
    value = random_choice(change_list[key])
    j = 0

    if key == TRANSPOSE:
        while value[0]+value[2] not in word:
            j += 1
            if j > 50:
                return word.replace(random_choice(change_list[DELETE]), '', 1)
            value = random_choice(change_list[key])
        return word.replace(value[0] + value[2], value[2] + value[0], 1)

    if key == INSERT:
        while value[0]+value[2] not in word:
            j += 1
            if j > 50:
                return word.replace(random_choice(change_list[DELETE]), '', 1)
            value = random_choice(change_list[key])
        itr_match = list(re.finditer(value[0]+value[2], word))
        to_change = random_choice(itr_match)
        return word[:to_change.span()[0]] + value + word[to_change.span()[1]:]

    if key == DELETE:
        while value not in word:
            j += 1
            if j > 50:
                return word
            value = random_choice(change_list[key])
        itr_match = list(re.finditer(value[0], word))
        to_change = random_choice(itr_match)
        return word[:to_change.span()[0]] + word[to_change.span()[1]:]

    if key == REPLACE:
        while value[0] not in word:
            j += 1
            if j > 50:
                return word.replace(random_choice(change_list[DELETE]), '', 1)
            value = random_choice(change_list[key])
        itr_match = list(re.finditer(value[0], word))
        to_change = random_choice(itr_match)
        return word[:to_change.span()[0]] + value[2] + word[to_change.span()[1]:]

    return word.replace(random_choice(change_list[DELETE]), 1)


def generate_noise_v2():
    noisy_word_list = []
    with open(ALPHABET, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        alphabet = first_line.split(',,,')

    with open(MANUAL_NOISE_FILE, 'r', encoding='utf-8') as f:
        changes = get_all_changes(f.readlines(), alphabet)      # {'T': [all transposes done], etc}
                                                                #
    distribution = ([TRANSPOSE] * len(changes[TRANSPOSE])) \
                   + ([REPLACE] * len(changes[REPLACE])) \
                   + ([INSERT]  * len(changes[INSERT])) \
                   + ([DELETE]  * len(changes[DELETE]))

    with open(NAME_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.lower().strip().split(',,,')
            noisy_word_list.append(',,,'.join([line[len(line)-1]] + line[1:]))
            if line[0] != line[len(line)-1]:
                noisy_word_list.append(',,,'.join(line))
            else:   # 66 % to steal change from manual-error, 33 % to create new
                # if rand() < 0.66:
                noisy_word_list.append(',,,'.join([manual_based_error(changes, distribution, line[len(line)-1])] + line[1:]))
                # else:
                #     noisy_word_list.append('%s,,,%s' % (keyboard_based_error(line[1]), line[1]))

    with open(AUTO_NOISE_FILE, 'w', encoding='utf-8') as f:
        shuffle(noisy_word_list)
        f.write('\n'.join(noisy_word_list))

    # noisy_word_list = []
    # with open(KORPUS_FILE, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         line = line.lower().strip().split(' ')
    #         noisy_word_list.append("%s,,,%s" % (line[0], line[0]))
    #         # 66 % to steal change from manual-error, 33 % to create new
    #         # if rand() < 0.66:
    #         noisy_word_list.append('%s,,,%s' % (manual_based_error(changes, distribution, line[0]), line[0]))
    #         # else:
    #         #     noisy_word_list.append('%s,,,%s' % (keyboard_based_error(line[0]), line[0]))
    #
    # with open(KORPUS_NOISED_FILE, 'w', encoding='utf-8') as f:
    #     shuffle(noisy_word_list)
    #     f.write('\n'.join(noisy_word_list))



if __name__ == '__main__':
    generate_noise_v2()
    # name_dict = {}
    # char_set = set()
    # with open(NAME_FILE, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         if ',,,' in line:
    #             street = line.split(',,,')[1].lower().strip()
    #             if len(street) > 1:
    #                 name_dict[street] = [street]
    #
    #
    # # char_set = [x for x in char_set]
    # for key in name_dict.keys():
    #     for x in range(0, 3):
    #         name_dict[key].append(generate_noisy_word(key, alphabet))
    #
    # with open(AUTO_NOISE_FILE, 'w', encoding='utf-8') as f:
    #     for key in name_dict.keys():
    #         for value in name_dict[key]:
    #             f.write("%s,,,%s\n" % (value, key))
