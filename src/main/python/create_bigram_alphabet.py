def get_bigrams(name):
    new_name = ''.join(['\t', name, '\n'])
    bigrams = []
    for i in range(len(new_name)-1):
        bigrams.append(new_name[i:i+2])
    return bigrams


if __name__ == '__main__': 
    with open('../../../data/nameDataUnique.csv', 'r') as f:
        with open('../../../data/bigramAlphabet.csv', 'w') as f2:
            bigrams = {}
            for line in f:
                split = line.strip().split(",,,")
                tmp = get_bigrams(split[1].lower())
                for bi in tmp:
                    if bi in bigrams:
                        bigrams[bi] += 1
                    else:
                        bigrams[bi] = 1
            f2.write(',,,'.join([t for t in bigrams if bigrams[t] > 5]))