with open('../../../data/nameDataUnique.csv', 'r') as f:
	with open('../../../data/speedData.csv', 'w') as f2:
		data = {
			'30':[],
			'70':[],
			'130':[]
		}
		for line in f:
			split = line.strip().split(",,,")
			if len(split) == 3 and split[2].isdigit() and split[2] != "-1":
				tmp = ''
				if int(split[2]) <= 30:
					tmp = ',,,'.join(split[:2] + ['30\n'])
					data['30'] += [tmp]
				elif int(split[2]) <= 70:
					tmp = ',,,'.join(split[:2] + ['70\n'])
					data['70'] += [tmp]
				else:
					tmp = ',,,'.join(split[:2] + ['130\n'])
					data['130'] += [tmp]
				f2.write(tmp)
		print(len(data['30']), " ", len(data['70']), " ", len(data["130"]))