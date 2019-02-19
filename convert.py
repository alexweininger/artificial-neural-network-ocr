

# converts the first column of the .data from characters to ints from 0-25, where A = 0, and Z = 25
def convertCharToInt():
	f = open('../original files/letter-recognition.data', 'r')
	out = open('./out.csv', 'w')
	for line in f.readlines():
		out.write(str(ord(line[0]) - 64) + line[1:])

# splits data up into training and validation data sets, each with size n
def splitData(n):
	f = open('./out.csv', 'r')
	t = open('./train.csv', 'w')
	v = open('./valid.csv', 'w')
	count = 0
	for line in f.readlines():
		if count < n:
			t.write(line)
		elif count < (n * 2):
			v.write(line)
		else:
			break
		count += 1

maxes = []
values = []
i = 0



convertCharToInt()
splitData(5000)
