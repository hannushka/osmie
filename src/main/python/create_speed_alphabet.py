with open('../../../data/nameDataUnique.csv','r') as f:
    speed = set()
    for line in f:
        sp = line.strip().split(",,,")
        if len(sp) == 3:
            speed.add(sp[2])
    with open('../../../data/speedAlphabet.csv', 'w') as f2:
        f2.write(',,,'.join(speed))