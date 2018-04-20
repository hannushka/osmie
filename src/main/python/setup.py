import subprocess
from smart_open import smart_open
import json
import pip
import regex as re
import operator
import os
from zipfile import ZipFile
#from scripts.auto_noise import generate_noisy_word

# ATLAS_OLD = "https://drive.google.com/open?id=1FQ2bG7T2ijVMqk9FpMiu-U9K4eX1l-ZE"
# ATLAS = "https://drive.google.com/open?id=1cA3VjKf0vShO6lTDAGEcAHHE1pUyTipw"
# Online sources
DK_WIKI_DUMP_URL = "https://dumps.wikimedia.org/etwiki/latest/etwiki-latest-pages-meta-current.xml.bz2"
DK_WIKI_DUMP = "etwiki-latest-pages-meta-current.xml.bz2"
# DA_HUNSPELL_DICT = "https://github.com/elastic/hunspell/raw/master/dicts/da_DK/da.dic"
# DK_KORPUS = "http://korpus.dsl.dk/resources/corpora/KDK-2010.scrambled.zip"

# Local sources
# DK_LOCAL_H_DICT = "data/da.dic"
# DK_LOCAL_CORPUS = "data/KDK-2010.scrambled"
DK_JSON = "dk_wiki.json.gz"
DK_TXT = 'data/dk_wiki.txt'
DK_WIKI_FREQ_DICT = 'data/wiki_freq_dict.txt'
# DK_KORPUS_FREQ_DICT = 'data/korpus_freq_dict.txt'
DK_ALPHABET = 'data/dk_alphabet.txt'
AUTO_NOISE_FILE = 'data/korpus_freq_dict.txt.mini.noised'


def remove_trash(line):
    return re.sub('[*(\'\'\')\(,\)"(\'\')\.=:|]', ' ', line).lower()


def generate_freq_dict():
    freq_dict = {}
    with open(DK_TXT, 'r') as f:
        f.readline()
        for line in f:
            clear_line = remove_trash(line)
            words = re.findall("\p{L}+", clear_line)
            for word in words:
                if word in freq_dict.keys():
                    freq_dict[word] += 1
                else:
                    freq_dict[word] = 1

    sorted_freq_dict = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
    with open(DK_WIKI_FREQ_DICT, 'w') as f:
        for item in sorted_freq_dict:
            if item[1] > 1:
                f.write(item[0] + " " + str(item[1]) + '\n')


def generate_corpus_dicts():
    word_dict = {}
    for dir in next(os.walk(DK_LOCAL_CORPUS))[1]:
        for file in next(os.walk(DK_LOCAL_CORPUS + '/' + dir))[2]:
            if 'txt' not in file:
                continue
            with open("%s/%s/%s" % (DK_LOCAL_CORPUS, dir, file), 'r') as f:
                for line in f:
                    if line.startswith('<'):
                        continue
                    word = re.match("\p{L}+", line)
                    if word:
                        word = word.group()
                        if word in word_dict.keys():
                            word_dict[word] += 1
                        else:
                            word_dict[word] = 1
    with open(DK_KORPUS_FREQ_DICT, 'w') as f:
        for key in word_dict.keys():
            if word_dict[key] > 2:
                f.write(key + " " + str(word_dict[key]) + '\n')


def generate_txt_file():
    print("Generating huge wiki txt file")
    for line in smart_open('data/%s' % DK_JSON):
        article = json.loads(line.decode('utf8'))
        article_text = article['title']

        for section_title, section_text in zip(article['section_titles'], article['section_texts']):
            article_text = ' '.join([article_text, section_title, section_text])

        with open(DK_TXT, 'a') as f:
            f.write(article_text)


def extract_corpus(pw):
    print("Extracting zip-file")
    with ZipFile(DK_LOCAL_CORPUS + ".zip", "r") as f:
        ZipFile.extractall(f, path="data/", pwd=pw.encode())


def download_corpus_and_generate_json():
    print("Downloading..")
    subprocess.call(["wget", DK_WIKI_DUMP_URL], cwd="data/")
    # subprocess.call(["wget", DK_KORPUS], cwd="data/")
    print("Extracting to json.gz..")
    subprocess.call(["python3", "-m", "gensim.scripts.segment_wiki", "-f",
                     DK_WIKI_DUMP, "-o", DK_JSON], cwd="data/")
    generate_txt_file()


def cleanup():
    print("Cleaning up..")
    subprocess.call(["rm", DK_WIKI_DUMP], cwd="data/")
    subprocess.call(["rm", DK_JSON], cwd="data/")
    subprocess.call(["rm", "-r", DK_LOCAL_CORPUS])
    subprocess.call(["rm", DK_LOCAL_CORPUS + ".zip"])


def generate_alphabet():
    alpha = {}
    with open(DK_WIKI_FREQ_DICT, 'r') as f:       # Can easily be changed to streets or wiki dump!
        for line in f:
            for char in line:
                if char.isdigit():
                    continue
                if char in alpha.keys():
                    alpha[char] += 1
                else:
                    alpha[char] = 1
    del alpha['\n']

    with open(DK_ALPHABET, 'w') as f:
        f.write(',,,'.join([key for key in alpha.keys() if alpha[key] > 500]))

# def create_minimal_n_noisy_corpus():
#     with open(DK_KORPUS_FREQ_DICT + ".mini", 'w') as fWrite:
#         with open(DK_KORPUS_FREQ_DICT, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 line = line.split()
#                 if len(line) > 0 and int(line[1]) > 25:
#                     fWrite.write('%s %s\n' %(line[0], line[1]))
#
#     char_set = set()
#     with open(DK_KORPUS_FREQ_DICT + ".mini", 'r', encoding='utf-8') as f:
#         for line in f:
#             if ' ' in line:
#                 street = line.split(' ')[0].lower().strip()
#                 if len(street) > 1:
#                     name_dict[street] = [street]
#                     for x in street:
#                         if re.match('\p{L}', x):#not x.isdigit() and x not in string.punctuation:
#                             char_set.add(x)
#
#     char_set = [x for x in char_set]
#     for key in name_dict.keys():
#         for _ in range(0, 3):
#             name_dict[key].append(generate_noisy_word(key, char_set))
#
#     with open(AUTO_NOISE_FILE, 'w', encoding='utf-8') as f:
#         for key in name_dict.keys():
#             for value in name_dict[key]:
#                 f.write("%s,,,%s\n" % (value, key))


if __name__ == '__main__':
    #if not os.path.exists(DK_LOCAL_CORPUS + ".zip") or not os.path.exists(DK_TXT):
    # download_corpus_and_generate_json()
    # if not os.path.exists(DK_LOCAL_CORPUS + "/01") and not os.path.exists(DK_KORPUS_FREQ_DICT):
    #     pw = input("DK Korpus requires PW to unload, please provide\n> ")
    #     if pw:
    #         extract_corpus(pw.strip())
    #         print("Generating korpus_freq_dict")
    #         generate_corpus_dicts()
    #         create_minimal_n_noisy_corpus()
    #
    #     else:
    #         print("No PW provided, shutting down.")
    #         quit()
    if not os.path.exists(DK_WIKI_FREQ_DICT):
        print("Generating wiki_freq_dict")
        # generate_freq_dict()
    generate_alphabet()
    # cleanup()