import csv
import math
import sys



def cap(array):
	numrows=0
	energies=[]
	capacity=0
	rounding=-1
	numpoints=0
	numclass1=0
	class1=''

	if 1==1:
		
		for row in array:
			numpoints=numpoints+1
			result = 0
			numrows=len(row[:-1])
			for elem in row[:-1]:
				result = result + float(elem) 
			c = row[-1]
			if (class1==''):
				class1=c
			if (c==class1):
				numclass1=numclass1+1
			if (rounding!=-1):
				result=int(result*math.pow(10,rounding))/math.pow(10,rounding)
			energies=energies+[(result, c)]
	sortedenergies=sorted(energies, key=lambda x: x[0])
	curclass=sortedenergies[0][1]
	changes=0
	for item in sortedenergies:
		if (item[1]!=curclass):
			changes=changes+1
			curclass=item[1]

	clusters=changes+1
	mincuts=math.ceil(math.log(clusters)/math.log(2))
	capacity=mincuts*numrows
	#tmlpcap=mincuts*(numrows+1)+(mincuts+1)
	print(mincuts)
	# The following assume two classes!
	entropy=-((float(changes)/numpoints)*math.log(float(changes)/numpoints)+(float(numpoints-changes)/numpoints)*math.log(float(numpoints-changes)/numpoints))/math.log(2)

	#print "Input dimensionality: ", numrows, ". Number of points:", numpoints, ". Class balance:", float(numclass1)/numpoints 
	#print "Eq. energy clusters: ", clusters, "=> binary decisions/sample:", entropy
	print ("Max capacity need: ", (changes*(numrows+1))+changes,"bits")
	print ("Estimated capacity need: ",int(math.ceil(capacity)),"bits")
	return int(math.ceil(capacity))