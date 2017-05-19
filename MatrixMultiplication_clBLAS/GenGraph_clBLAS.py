#Script to generate the necessary plots 
#by making use of the stored results obtained after 
#executing the MatrixMul_Test_clBLAS.py 
import matplotlib

#enable this if the other one fails
#matplotlib.use('Agg')
matplotlib.use('GtkAgg') 

import matplotlib.pyplot as plt


import time
import sys
sys.path.insert(0,'./cpp')
from pylab import *
import pickle
import argparse
from scipy import stats

import numpy


parser = argparse.ArgumentParser(description='Generates a graph to visuallize the Data')
parser.add_argument('--NolinChange',action='store_true',help='The dissabels the different styles (solid dotted etc.) that are used to draw the lines (colors are still different)')
parser.add_argument('--NoPlot',action='store_true',help='Will not show/generate the BoxPlot (auto enabled for 2d Plots)')
parser.add_argument('--NoLines',action='store_true',help='Will not show/generate the 1D/2D line Plots (does not affect BoxPlots)')
parser.add_argument('--Show',action='store_true',help='Displays the Results directly on Screan without saving them to files')
parser.add_argument('--BaseScale',action='store_true',help='Will automaticaly generate the best scale ignoring the expected')
parser.add_argument('--InitialSkip', nargs='?', type=int, default=500,help='Allows to ignore the first N results from the Files (usefull if the first N files are slower or have other problems) (Does not affect Non Heurestric data like Register used)')
parser.add_argument('--plotargs', nargs='*', type=int,help='The dimensions that you want to display (example -1 6 -1 -> Transforms the 3D input data into a 2D input data where the secound dimension is 6 2D plot ,example 5 -1 3 -> displays a graph with dimension 1 set to 5 and dimension 3 set to 3) (Ignored in 1D graphs)')
parser.add_argument('--infiles', nargs='+', type=argparse.FileType('r'),help='The files generated that you want to compare or visuallise')
args = parser.parse_args()

colors = ["blue", "red", "green","black","yellow"]
if args.NolinChange :
	linestyls =['-b' , '-b', '-b', '-b', '-b', '-b']
	linethikness =[ 1,1,1,1,1,1 ]
else:
	linestyls =['-b' , '--', '-.', '..']
	linethikness =[ 3,4,5,6 ]

throwaway = True

firstRun = (args.plotargs[:1*len(args.plotargs)]).count(-1)

d2plot = False

d0plot = False

#switch
if firstRun == 0:
	args.NoLines = True
	d0plot = True
else:
	if firstRun != 1:
		if firstRun == 2:
			d2plot = True
		else:
			print "the Number of Dimenstions to display must be less or equal 2"
			sys.exit(1)

res = []
for readfile in args.infiles:
	res.append(pickle.load(readfile))
	readfile.close()

def MorthList(dataArg):
	if isinstance(dataArg[0],list):
		for i in xrange(len(dataArg)):
			dataArg[i] = MorthList(dataArg[i])
		return dataArg
	else:
		return dataArg[args.InitialSkip:]

#throw the first Sampels away they can be faulty
#if res[0]["IsHeurestric"]:
#	for j in xrange(len(args.infiles)):
#		for k in xrange(len(res[j]["Datay"])):
#			for l in xrange(len(res[j]["Datay"][k])):
#				res[j]["Datay"][k][l] = res[j]["Datay"][k][l][args.InitialSkip:]

IsHeurestric = True
if ('IsHeurestric' in res[0]):
	IsHeurestric = res[0]["IsHeurestric"]
Depth = 1
if ('Depth' in res[0]):
	Depth = res[0]["Depth"]


if IsHeurestric:
	for j in xrange(len(args.infiles)):
		res[j]["Datay"] = MorthList(res[j]["Datay"])

if not IsHeurestric:
	print "non heurestric Data"
	args.NoPlot = True

if (d2plot == True):
	args.NoPlot = True



def getLinea(dataArg,info,useinfo):
	if len(useinfo) == 0:
		return dataArg
	else:
		if useinfo[0] == 0:
			return getLinea(dataArg[info[0]],info[1:],useinfo[1:])
		else:
			res = []
			for i in dataArg:
				res.append(getLinea(i,info[1:],useinfo[1:]))
			return res

def getLine(dataArg,info):
	useinfo = []	
	for i in xrange(len(info)):
		if info[i] == -1:
			useinfo.append(1)
		else:
			useinfo.append(0)
	return getLinea(dataArg,info,useinfo)

def getNindex(liste,data,inde):
	curf = 0
	i = 0
	for j in liste:
		if data == j:
			if curf == inde:
				return i
			curf = curf + 1
		i = i + 1


i = 0

if Depth == 1:
	rangeLen = 1
else:
	rangeLen = len(args.plotargs)/Depth

for data in res[0]["DataNames"]:
	if args.NoPlot == False :
		#process all the needed stuff
		for m in xrange(rangeLen):
			for j in xrange(len(args.infiles)):
				if d0plot == True:
					currentDatay = getLine(res[j]["Datay"][i],args.plotargs[m*len(args.plotargs):(m + 1)*len(args.plotargs)])
					figname = str(data) + res[j]["Name"]
					f=figure(figsize=[10,5],num = figname)
					density = stats.kde.gaussian_kde(currentDatay)
					z_min, z_max = min(currentDatay), max(currentDatay)
					x = numpy.arange(z_min, z_max, (z_max - z_min) /1000.0)
					plot(x,density(x))
					if args.Show == True:
						f.show()
					else:
						f.savefig(str(data) + res[j]["Name"] + '.png')
				else:
					if Depth == 1:
						currentDatay = res[j]["Datay"][i]
					else:
						currentDatay = getLine(res[j]["Datay"][i],args.plotargs[m*len(args.plotargs):(m + 1)*len(args.plotargs)])
					figname = str(data) + res[j]["Name"]
					f=figure(figsize=[10,5],num = figname)
					poslist = []
					for m in xrange(len(currentDatay)):
						poslist.append(m);
					boxplot(currentDatay,positions=poslist,notch = True)
					if (args.BaseScale == False):
						ylim((0,res[0]["BaseScalyMax"][i]))
					if args.Show == True:
						f.show()
					else:
						if Depth == 1:
							f.savefig(str(data) + res[j]["Name"] + '.png')
						else:
							f.savefig(str(data) + res[j]["Name"] + str(m) + '.png')
	
	if args.NoLines == False:
		
		#process all the needed stuff
		for m in xrange(rangeLen):
			for j in xrange(len(args.infiles)):
				f=figure(figsize=[10,5],num = data)
				if (d2plot == True):
					currentDatay = getLine(res[j]["Datay"][i],args.plotargs[m*len(args.plotargs):(m + 1)*len(args.plotargs)])
					temp1 = getNindex(args.plotargs[m*len(args.plotargs):(m + 1)*len(args.plotargs)],-1,0)
					currentDatax1 = res[j]["Datax"][i][temp1]
					temp1 = getNindex(args.plotargs[m*len(args.plotargs):(m + 1)*len(args.plotargs)],-1,1)
					currentDatax2 = res[j]["Datax"][i][temp1]

					temp = []
					for k in currentDatay:
						temp2 = []
						for q in k:
							if IsHeurestric:
								temp2.append(average(q))
							else:
								temp2.append(q)
						temp.append(temp2)
					resA = np.array(temp)
					z_min, z_max = resA.min(), resA.max()
					AcurrentDatax1 = np.array(currentDatax1)
					AcurrentDatax2 = np.array(currentDatax2)
					plt.pcolor(AcurrentDatax2, AcurrentDatax1, resA, cmap='RdBu', vmin=z_min, vmax=z_max)
					plt.axis([AcurrentDatax1.min(), AcurrentDatax1.max(), AcurrentDatax2.min(), AcurrentDatax2.max()])
					cb = plt.colorbar()
					if args.Show == True:
						plt.show()
					else:
						plt.savefig(data + res[j]["Name"] + str(m) + '.png')
						plt.clf()						
						plt.cla()
						plt.close()
				else:
					if Depth == 1:
						currentDatay = res[j]["Datay"][i]
						currentDatax = res[j]["Datax"][i]
					else:
						currentDatay = getLine(res[j]["Datay"][i],args.plotargs[m*len(args.plotargs):(m + 1)*len(args.plotargs)])
						temp1 = getNindex(args.plotargs[m*len(args.plotargs):(m + 1)*len(args.plotargs)],-1,0)
						currentDatax = res[j]["Datax"][i][temp1]

					temp = []
					for k in currentDatay:
						if IsHeurestric:
							temp.append(average(k))
						else:
							temp.append(k)
					plot(currentDatax,temp,linestyls[j],label=res[j]["Name"],color=colors[j],linewidth=linethikness[j] )
					plot(currentDatax,temp,'r.')
			if (d2plot == False):
				if (args.BaseScale == False):
					ylim(0,res[0]["BaseScalyMax"][i])
				xlabel(res[0]["xName"][i])
				ylabel(res[0]["yName"][i])
				legend(loc='best')
				if args.Show == True:
					f.show()
				else:
					f.savefig(data + str(m) + '.png')
	i = i + 1
if ((args.Show == True) & (d2plot == False)): #keep the plots open
	print("Keep the Programm running to keep the Plots open press strg+C to close the Programm and all Graphs")
	pause(2**31 - 1)

