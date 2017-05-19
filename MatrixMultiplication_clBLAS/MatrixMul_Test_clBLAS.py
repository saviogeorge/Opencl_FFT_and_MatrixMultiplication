#This script drives the MMUL_clBLAS.cpp program 
#it tests the clBLAS for The application was 
#tested for different sizes of input matrix. 
#Each combination of the input matrices was 
#tested for 1000 times with random float value. 
import time
import pickle
import sys
sys.path.insert(0,'./cpp')

from pylab import *
import MMUL_clBLAS_Lib
import random
import numpy

#This variable decides
#how manytimes C=AxB has to carried out
#with different values 
AVGNumber = 1000

#The following variables decides the
#sizes of the Matrices
beginK = 16
endK = 32 + 1
stepK = 8

beginM = 160*1
endM = 160*7 + 1
stepM = 160*1

beginN = 160*1
endN = 160*7 + 1
stepN = 160*1


x = []

temp = []
for N in xrange(beginN,endN,stepN):
	temp.append(N)
x.append(temp)

temp = []
for M in xrange(beginM,endM,stepM):
	temp.append(M)
x.append(temp)

temp = []
for K in xrange(beginK, endK,stepK):
	temp.append(K)
x.append(temp)

y =  [[[0 for k in xrange((endK - beginK)/stepK + 1)] for j in xrange((endM - beginM)/stepM + 1)] for i in xrange((endN - beginN)/stepN + 1)]
y2 = [[[0 for k in xrange((endK - beginK)/stepK + 1)] for j in xrange((endM - beginM)/stepM + 1)] for i in xrange((endN - beginN)/stepN + 1)]
y3 = [[[0 for k in xrange((endK - beginK)/stepK + 1)] for j in xrange((endM - beginM)/stepM + 1)] for i in xrange((endN - beginN)/stepN + 1)]
y4 = [[[0 for k in xrange((endK - beginK)/stepK + 1)] for j in xrange((endM - beginM)/stepM + 1)] for i in xrange((endN - beginN)/stepN + 1)]
y5 = [[[0 for k in xrange((endK - beginK)/stepK + 1)] for j in xrange((endM - beginM)/stepM + 1)] for i in xrange((endN - beginN)/stepN + 1)]
y6 = [[[0 for k in xrange((endK - beginK)/stepK + 1)] for j in xrange((endM - beginM)/stepM + 1)] for i in xrange((endN - beginN)/stepN + 1)]

OpenlClBLASObj = MMUL_clBLAS_Lib.OpenClMMUL_clBLAS()

for K in xrange(beginK, endK,stepK):
	for M in xrange(beginM,endM,stepM):
		for N in xrange(beginN,endN,stepN):
			err = []
			Exetimem = []
			Writetime = []
			Readtime = []
			OpenlClBLASObj.SetParam(M,K,N)
			for j in xrange(0,AVGNumber):
				m1 = [[random.random() + 0.5 for te1 in range(K)] for te2 in range(M)]
				m2 = [[random.random() + 0.5 for te1 in range(N)] for te2 in range(K)]
							
				m1 = np.array(m1,dtype=float32)
				m2 = np.array(m2,dtype=float32)
				ref = [[0 for k in xrange(M)] for j in xrange(N)]

				ref = numpy.dot(m1,m2)

				res = OpenlClBLASObj.doMMUL(m1,m2)

				Writetime.append(OpenlClBLASObj.getWriteTime())
				Exetimem.append(OpenlClBLASObj.getExeTime())
				Readtime.append(OpenlClBLASObj.getReadTime())

				toerr = np.average(abs((ref - res) / (ref)))
				err.append(toerr)

			Ni = (N - beginN)/stepN
			Mi = (M - beginM)/stepM
			Ki = (K - beginK)/stepK

			y[Ni][Mi][Ki] = Writetime
			y2[Ni][Mi][Ki] = Exetimem
			y3[Ni][Mi][Ki] = Readtime
			y4[Ni][Mi][Ki] = err

			print M, K, N

d = {"Name" : sys.argv[1], #The name of the thing that got messured
"DataNames" : ["WriteTime","ExecuteTime","ReadTime","Discrepancy"], #The name of the Results
"Datay" : [y,y2,y3,y4], #The raw Data
"Datax" : [x,x,x,x],
"xName" : ["N in points","N in points","N in points","N in points"],
"yName" : ["Time in ms","Time in ms","Time in ms",""],
"BaseScalyMax" : [1,20,1,1e-7],
"IsHeurestric" : True,
"Depth" : 3} #The raw Data


afile = open(r'results', 'wb')
pickle.dump(d,afile)
afile.close()


