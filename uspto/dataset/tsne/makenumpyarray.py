import numpy, sys
from rdkit import Chem
from rdkit.Chem import AllChem

radius = 3
nbits = 4096 #2048 #1024 #512

fps = []
for line in open(sys.argv[1]):
    b, x = line.strip().split()
    bmol = Chem.MolFromSmiles(b)
    xmol = Chem.MolFromSmiles(x)
    bdict = dict()
    _ = AllChem.GetMorganFingerprintAsBitVect(bmol, radius=radius, nBits=nbits, useFeatures=False, bitInfo=bdict)
    xdict = dict()
    _ = AllChem.GetMorganFingerprintAsBitVect(xmol, radius=radius, nBits=nbits, useFeatures=False, bitInfo=xdict)
    blist = [ len(bdict.get(x,[])) for x in range(nbits)]
    xlist = [ len(xdict.get(x,[])) for x in range(nbits)]
    bxlist = blist + xlist
    fps.append( bxlist)

ar = numpy.array(fps)
with open(sys.argv[1]+'.npy', 'wb') as fw:
    numpy.save(fw, ar)
print( ar.shape)