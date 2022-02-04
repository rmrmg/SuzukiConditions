import numpy
import matplotlib
import matplotlib.pyplot as plt

data = numpy.load( open('uspto_reaxys_tsne2d.npy', 'rb'))
matplotlib.rcParams.update({'font.size': 18})
x1 = data[:9586,0]
y1 = data[:9586,1]
x2 = data[9586:,0]
y2 = data[9586:,1]
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1, 1, 1)
color1 = ['#A9D18E' for i in range(9586)] #uspto
color2 = ['#A86ED4' for i in range(11860)]
a1 = [1 for i in range(9586)] #uspto
a2 = [0.2 for i in range(11860)]
l1 = ['USPTO' for i in range(9586)]
l2 = ['reaxys' for i in range(11860)]
#clr = numpy.array( color1+color2)
clr = color1+color2
#ax.scatter(x, y,c=clr, alpha=a1+a2, marker=',', s=15, label=l1+l2)
ax.scatter(x1, y1, c='#A9D18E', alpha=1, marker=',', s=11, label='USPTO')
ax.scatter(x2, y2, c='#A86ED4', alpha=0.2, marker=',', s=11, label='Reaxys')
ax.legend()
#leg = ax.get_legend()
#leg.legendHandles[0].set_color('red')
#leg.legendHandles[1].set_color('yellow')
#plt.figure(figsize=(6,4))
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('comperison.jpg')