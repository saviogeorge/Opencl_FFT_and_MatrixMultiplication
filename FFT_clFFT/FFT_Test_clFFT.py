#This script drives the FFT_clFFT.cpp program 
#it tests the clFFT for 2^n sizes (where n= 10, 11, 12, 13, 14) data sets
#Each set of data was tested for a 5000 times (AVGNumber) with random float values
#the results are stored in the form of a file sing pickle
#read time, write time, Kernel Execution time and the computed FFT discrepancy values
import time
import sys
sys.path.insert(0,'./cpp')
import numpy as np
import FFT_clFFT
import pickle

begin = 10
end = 15
AVGNumber = 5000

x = list(range(begin,end))
y = list(range(begin,end))
y2 = list(range(begin,end))
y3 = list(range(begin,end))
y4 = list(range(begin,end))

OpenlClFFTObj = FFT_clFFT.OpenClFFT();

for i in xrange(begin, end):
	currentN = 2 ** i
	OpenlClFFTObj.SetParam(currentN)
	RawList1 = []
	RawList2 = []
	RawList3 = []
	RawList4 = []
	for j in xrange(0,AVGNumber):
		a = np.random.random_sample(currentN,) -0.5
		a = np.array(a,dtype=np.complex64)
		c = np.fft.fft(a)
		res = OpenlClFFTObj.doFFT(a)
		
		RawList1.append(OpenlClFFTObj.getWriteTime())
		RawList2.append(OpenlClFFTObj.getExeTime())
		RawList3.append(OpenlClFFTObj.getReadTime())
		RawList4.append(np.average(abs(c - res) / abs(c)))

	y[i - begin] = RawList1
	y2[i - begin] = RawList2
	y3[i - begin] = RawList3
	y4[i - begin] = RawList4
	x[i - begin] = currentN
	OpenlClFFTObj.ResetParam()

#save the result in a file
d = {"Name" : sys.argv[1], #The name of the thing that got messured
"DataNames" : ["WriteTime","ExecuteTime","ReadTime","Discrepancy"], #The name of the Results
"Datay" : [y,y2,y3,y4], #The raw Data
"Datax" : [x,x,x,x],
"xName" : ["N in points","N in points","N in points","N in points"],
"yName" : ["Time in ms","Time in ms","Time in ms",""],
"BaseScalyMax" : [0.15,0.14,0.15,1e-6],
"IsHeurestric" : True,
"Depth" : 1} #The raw Data

afile = open(r'results', 'wb')
pickle.dump(d,afile)
afile.close()

