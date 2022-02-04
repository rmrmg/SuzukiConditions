import numpy
import matplotlib.pyplot as plt

data = numpy.load( open('uspto_reaxys_tsne2d.npy', 'rb'))

x = data[:,0]
y = data[:,1]
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
color1 = ['#7CAFDE' for i in range(5434)] #uspto
color2 = ['#A86ED4' for i in range(11860)]
a1 = [1 for i in range(5434)] #uspto
a2 = [0.01 for i in range(11860)]
#clr = numpy.array( color1+color2)
clr = color1+color2
ax.scatter(x, y,c=clr, alpha=a1+a2, marker=',', s=15)

a1 = [0.01 for i in range(5434)] #uspto
a2 = [1 for i in range(11860)]
bx = fig.add_subplot(1, 2, 2)
bx.scatter(x, y,c=clr, alpha=a1+a2, marker=',', s=15)

#plt.legend(loc='upper left')
plt.show()