import numpy
from sklearn.manifold import TSNE
uspto = numpy.load(open('uspto.bx.npy', 'rb') )
reaxys = numpy.load( open('reaxsys.bx.npy', 'rb'))
tsne = TSNE(n_components=2,  init='pca', perplexity=30, )

both = numpy.vstack([uspto, reaxys])
print(uspto.shape, reaxys.shape, both.shape)
wid = int(both.shape[1]//2)
print("W", wid )
#both1 = both[]
#both=[]
casted = tsne.fit_transform(both[:,:wid])
print(casted.shape)
with open('uspto_reaxys_tsne2d_p1.npy', 'wb') as fw:
    numpy.save(fw, casted)

casted = tsne.fit_transform(both[:,wid:])
print(casted.shape)
with open('uspto_reaxys_tsne2d_p2.npy', 'wb') as fw:
    numpy.save(fw, casted)
