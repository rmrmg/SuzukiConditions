import numpy
from sklearn.manifold import TSNE
uspto = numpy.load(open('uspto.bx.npy', 'rb') )
reaxys = numpy.load( open('reaxsys.bx.npy', 'rb'))
tsne = TSNE(n_components=2,  init='random')
both = numpy.vstack([uspto, reaxys])
shape = both.shape
for i in range( shape[1]):
    print(i, numpy.sum(both[:,i]))



