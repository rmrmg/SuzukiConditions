import numpy
import matplotlib
import matplotlib.pyplot as plt

data = numpy.load( open('uspto_reaxys_tsne2d.npy', 'rb'))
matplotlib.rcParams.update({'font.size': 18})
x1 = data[:5434,0]
y1 = data[:5434,1]
x2 = data[5434:,0]
y2 = data[5434:,1]
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1, 1, 1)

ax.scatter(x1, y1, c='#A9D18E', alpha=1, marker=',', s=11, label='USPTO')
ax.scatter(x2, y2, c='#A86ED4', alpha=0.2, marker=',', s=11, label='Reaxys')
ax.legend()

plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('comperison.jpg')