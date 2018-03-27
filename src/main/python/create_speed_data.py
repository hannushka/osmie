with open('../../../data/nameDataUnique.csv', 'r') as f:
	with open('../../../data/speedData.csv', 'w') as f2:
		data = {
			'0':[],
			'1':[],
			'2':[]
		}
		for line in f:
			split = line.strip().split(",,,")
			if len(split) == 3 and split[2].isdigit() and split[2] != "-1":
				tmp = ''
				if int(split[2]) <= 30:
					tmp = ',,,'.join(split[:2] + ['0\n'])
					data['0'] += [tmp]
				elif int(split[2]) <= 70:
					tmp = ',,,'.join(split[:2] + ['1\n'])
					data['1'] += [tmp]
				else:
					tmp = ',,,'.join(split[:2] + ['2\n'])
					data['2'] += [tmp]
				f2.write(tmp)
		print(len(data['0']), " ", len(data['1']), " ", len(data["2"]))