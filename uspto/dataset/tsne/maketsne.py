import numpy
from sklearn.manifold import TSNE
uspto = numpy.load(open('uspto.bx.npy', 'rb') )
reaxys = numpy.load( open('reaxsys.bx.npy', 'rb'))
tsne = TSNE(n_components=2,  init='pca', perplexity=30, )

both = numpy.vstack([uspto, reaxys])
print(uspto.shape, reaxys.shape, both.shape)
wid = both.shape[1]
casted = tsne.fit_transform(both)
print(casted.shape)
with open('uspto_reaxys_tsne2d.npy', 'wb') as fw:
    numpy.save(fw, casted)


