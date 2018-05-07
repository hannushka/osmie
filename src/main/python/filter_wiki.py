import os
import regex as re

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.rsplit('\\', 3)[0]
WIKI_UNFILTERED = '%s\data\dk_wiki.txt' % dir_path
WIKI_FILTERED = '%s\data\dk_wiki.filtered.txt' % dir_path
WIKI_ALPH = "%s\data\wiki_alph.txt" % dir_path


def replace_unwanted(line, char_dict):
    new_word = ""
    for c in line:
        if char_dict[c] > 10000:
            new_word += c
    return new_word

def gen_alphabet():
    chars = set()
    with open(WIKI_FILTERED, 'r', encoding="UTF-8") as f:
        for line in f:
            for c in line:
                chars.add(c)
    with open(WIKI_ALPH, 'w', encoding='UTF-8') as f:
        f.write(',,,'.join(list(chars)))

if __name__ == '__main__':
    gen_alphabet()
    quit()
    list_of_char = dict()
    with open(WIKI_UNFILTERED, 'r', encoding="UTF-8") as f:
        for line in f:
            for c in line:
                if c in list_of_char.keys():
                    list_of_char[c] += 1
                else:
                    list_of_char[c] = 1

    with open(WIKI_UNFILTERED, 'r', encoding="UTF-8") as f:
        with open(WIKI_FILTERED, 'w', encoding="UTF-8") as fWrite:
            for line in f:
                line = re.sub("'{2,}", "", line)
                line = line.replace(':', ' ')
                line = line.replace('|', ' ')
                line = line.replace('*', '')
                line = line.replace("'+", "")
                line = line.replace("=", "")
                line = line.replace("#", "")
                line = line.replace(";", " ")
                line = line.replace(", ,", ", ")
                line = replace_unwanted(line, list_of_char)
                line = line.replace("()", "")
                line = line.replace("\s+", " ")
                line = line.strip()

                if len(line) > 5 and len(line.split()) > 4:
                    fWrite.write(line + '\n')
