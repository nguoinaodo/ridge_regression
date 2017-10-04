import csv
import os

def readCsv(filepath):
	file = open(filepath)
	result = csv.reader(file, delimiter = ',')
	rows = []
	for r in result:
		rows.append(r)
	return rows

train = readCsv('data/1-prostate-training-data.csv')
test_predict = readCsv('data/20142776-test.csv')