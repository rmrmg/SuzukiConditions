import numpy
from rdkit import Chem
from sklearn import preprocessing, decomposition, cluster, model_selection
from tensorflow import Graph
from keras.models import model_from_json
from rdkit.Chem import AllChem, Descriptors

def parse(lines):
    data=[]
    #["ONERX", ["benzoin6m1N"], [], ["benzoin6m1N:::Brc1cccc2ncccc12"], [], 
    #5 ["ethanol", "toluene", "water"], ["sodium carbonate"], [80], [], ["Pd[P(Ph)3]4"], ["Inert atmosphere"], 
    #11 [13.0], ["ARTICLE", "Article; De Vreese, Rob; Muylaas; Tetrahedron Letters; vol. 58; 40; (2017); p. 3803 - 3807;"], 
    #13: {"Cc1ccc(S(=O)(=O)Oc2ccc(-c3ccc(-c4ccc(OS(=O)(=O)c5ccc(C)cc5)c5ncccc45)s3)c3cccnc23)cc1": [["CC1=CC=C(C=C1)S(=O)(=O)OC1=CC=C(Br)C2=CC=CN=C12", "Cc1ccc(S(=O)(=O)Oc2ccc(-c3ccc(B(O)O)s3)c3cccnc23)cc1"]]}, "CC1=CC=C(C=C1)S(=O)(=O)OC1=CC=C(Br)C2=CC=CN=C12.OB(O)C1=CC=C(S1)B(O)O.CC1=CC=C(C=C1)S(=O)(=O)OC1=CC=C(Br)C2=CC=CN=C12>>CC1=CC=C(C=C1)S(=O)(=O)OC1=C2N=CC=CC2=C(C=C1)C1=CC=C(S1)C1=C2C=CC=NC2=C(OS(=O)(=O)C2=CC=C(C)C=C2)C=C1"]
    for line in lines:
        ev=eval(line)
        boro = set()
        halo = set()
        for prod in ev[13]:
            for sbsList in ev[13][prod]:
                halo.add(sbsList[0])
                boro.add(sbsList[1])
        if len(boro) >1:
            boro2=boro.difference(halo)
            if len(boro2) != len(boro) and boro2:
                boro=boro2
        if len(halo) >1:
            halo2= halo.difference(boro)
            if halo2 and len(halo2) != len(halo):
                halo = halo2
        data.append( {'solvent':ev[5], 'base':ev[6], 'temp':ev[7], 'yield':ev[11], 'boronic':boro, 'halogen':halo} )
    return data

def getBaseClass( b):

    if 'carbonate' in b:
        return 'carbonate'
    if 'hydroxide' in b:
        return 'hydroxide'
    if 'phosphate' in b:
        return 'phosphate'
    if 'acetate' in b:
        return 'acetate'
    if 'fluoride' in b:
        return 'fluoride'
    if 'amine' in b or b =='DBU' or b=='DABCO':
        return 'amine'
    if 'ethanolate' in b or 'butanolate' in b or 'NaOMe' == b:
        return 'alkoxide'
    return 'other'

def getSolventClass(fros):
    if len(fros) ==1:
        if 'alcohol' in fros or 'amide' in fros or 'water' in fros:
            return 'polar'
        #alcohols, polar solv/water, water/alcohols, water/amides, water, amides
        if 'aromatic' in fros:
            return 'aromatic'
        if 'etheric' in fros:
            return 'etheric'
    if len(fros) ==2:
        if 'water' in fros and ( 'ace' in fros or 'alcohol' in fros or 'amide' in fros):
            return 'polar'
        if 'aromatic' in fros and ('water' in fros or 'alcohol' in fros):
            return 'polarAromatic'
        if 'water' in fros and 'etheric' in fros:
            return 'waterEther'
    if len(fros) ==3:
        if 'water' in fros and 'alcohol' in fros and 'aroatic' in fros:
            return 'polarAromatic'
    return 'other'



def getSolventClassAP(s):
    if s in ('toluene', 'benzene', 'xylene', 'para-xylene', '1,3,5-trimethyl-benzene', 'o-xylene'):
        return 'aromatic'
    if s in ('ethanol', 'methanol', 'propan-1-ol', 'isopropyl alcohol'):
        return 'alcohol'
    if s in ( 'tetrahydrofuran', 'diethyl ether', '1,4-dioxane', '1,2-dimethoxyethane', 'tetrahydrofuran-d8', '1,3-dioxane'):
        return 'etheric'
    if s in ('N,N-dimethyl acetamide', 'N,N-dimethyl-formamide'):
        return 'amide'
    if s in ('butan-1-ol', ):
        return 'BuOH'
    if s in ('water-d2', 'aq. phosphate buffer', 'water'):
        return 'water'
    if s in ('ethylene glycol',):
        return 'EG'
    if s in ('acetonitrile', 'acetone'):
        return 'ace'
    if s in ('dichloromethane', ):
        return 'DCM'
    return 'other'


def makeOutput(base, solvent):
    baseDict = { frozenset({'acetate'}):6, frozenset({'amine'}):5, frozenset({'hydroxide'}):4, frozenset({'fluoride'}):3,
                frozenset({'phosphate'}):2, frozenset({'carbonate'}):1, 
    }
    
    solventDictOld = {  frozenset({'alcohol'}):12, frozenset({'ace', 'water'}):11, frozenset({'water', 'aromatic'}):10,  frozenset({'water', 'alcohol'}):9,
                frozenset({'water'}):8, frozenset({'amide'}):7,  frozenset({'alcohol', 'aromatic'}):6, frozenset({'aromatic'}):5,  frozenset({'water', 'amide'}):4,
                frozenset({'water', 'alcohol', 'aromatic'}):3,   frozenset({'etheric'}):2,  frozenset({'water', 'etheric'}):1,
    }
    solventDict = { 'polar':1, 'aromatic':2, 'etheric':3, 'polarAromatic':4, 'waterEther':5, 'other':6}
    return baseDict.get(base, len(baseDict)+1), solventDict.get(solvent, len(solventDict)+1)


def getRdkitDesc(smi):
    mol = Chem.MolFromSmiles(smi)
    return [ str(x[1](mol)) for x in Descriptors.descList]


def getIncoFP(smi, incoGr):
    mol=Chem.MolFromSmiles(smi)
    return [ str(sum([mol.HasSubstructMatch(gr) for gr in x ])) for x in incoGr]


if __name__ == "__main__":
    import sys
    lines=[]
    #mode='rdkit' #'enc'
    #mode = 'rdkit+morgan'
    #mode = 'plain' #+rdkit+morgan'
    #mode='morgan3' #+inco'
    mode='inputForGCNN'
    if mode == 'enc':
        haloEnc = {x.split('\t')[0]:x.strip().split('\t')[1:] for x in open('encodedHalogens.8d') }
        boroEnc = {x.split('\t')[0]:x.strip().split('\t')[1:] for x in open('encodedBoronicAcid.8d') }
    elif mode == 'plain':
        haloEnc={smi.strip():desc.strip().split('\t') for smi, desc in zip(open('opisSubstratow/halogenicSmiles').readlines() , open('opisSubstratow/halogens').readlines()) }
        boroEnc={smi.strip():desc.strip().split('\t') for smi, desc in zip(open('opisSubstratow/boronicAcidSmiles').readlines() , open('opisSubstratow/boronociAcidsInp').readlines()) }
    elif mode == 'morgan3':
        fplen=512
        haloEnc = {smi.strip():[x for x in AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(smi),3,nBits=fplen ).ToBitString()] for smi in open('opisSubstratow/halogenicSmiles').readlines()  }
        boroEnc = {smi.strip():[x for x in AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(smi),3, nBits=fplen ).ToBitString()] for smi in open('opisSubstratow/boronicAcidSmiles').readlines()  }
    elif mode == 'rdkit':
        haloEnc = {smi.strip():getRdkitDesc(smi) for smi in open('opisSubstratow/halogenicSmiles').readlines()  }
        boroEnc = {smi.strip():getRdkitDesc(smi) for smi in open('opisSubstratow/boronicAcidSmiles').readlines()  }
    elif mode == 'rdkit+morgan':
        fplen=2048
        haloEnc = {smi.strip():[x for x in AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(smi),3, nBits=fplen ).ToBitString()]+getRdkitDesc(smi) for smi in open('opisSubstratow/halogenicSmiles').readlines()  }
        boroEnc = {smi.strip():[x for x in AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(smi),3, nBits=fplen ).ToBitString()]+getRdkitDesc(smi) for smi in open('opisSubstratow/boronicAcidSmiles').readlines()  }
    elif mode == 'plain+rdkit+morgan':
        fplen=512
        haloEnc = {smi.strip():desc.strip().split('\t') + getRdkitDesc(smi)+ [x for x in AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(smi),3, nBits=fplen ).ToBitString()]
                for smi, desc in zip(open('opisSubstratow/halogenicSmiles').readlines() , open('opisSubstratow/halogens').readlines()) }
        boroEnc = {smi.strip():desc.strip().split('\t') + getRdkitDesc(smi)+ [x for x in AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(smi),3, nBits=fplen ).ToBitString()]
                for smi, desc in zip(open('opisSubstratow/boronicAcidSmiles').readlines() , open('opisSubstratow/boronociAcidsInp').readlines()) }
    elif mode =='morgan3+inco':
        fplen=512
        incoGr = [ [Chem.MolFromSmarts(m) for m in x.strip().split('.')] for x in  open('Grupy_niekompatybilne.csv') if x.strip()]
        haloEnc = {smi.strip():[x for x in AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(smi),3,nBits=fplen ).ToBitString()]+getIncoFP(smi, incoGr) 
                    for smi in open('opisSubstratow/halogenicSmiles').readlines()  }
        boroEnc = {smi.strip():[x for x in AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(smi),3, nBits=fplen ).ToBitString()]+getIncoFP(smi, incoGr) 
                    for smi in open('opisSubstratow/boronicAcidSmiles').readlines()  }

    elif mode == 'inputForGCNN':
        haloEnc = {smi.strip():[smi.strip(),] for smi in open('opisSubstratow/halogenicSmiles').readlines()  }
        boroEnc = {smi.strip():[smi.strip(),] for smi in open('opisSubstratow/boronicAcidSmiles').readlines()  }

    for fn in sys.argv[1:]:
        for line in open(fn):
            lines.append(line)
    dane = parse(lines)
    allBases=dict()
    allSolvent=dict()
    for i in dane:
        if len(i['boronic']) != 1 or len(i['halogen']) !=1:
            continue
        base = frozenset([getBaseClass(b) for b in i['base'] ]) 
        solv =getSolventClass(  frozenset([ getSolventClassAP(s) for s in i['solvent'] ]) )
        baseClass, solvClass = makeOutput(base ,solv)
        try:
            halo = tuple(i['halogen'])[0]
            halodata= haloEnc[halo]
        except:
            halodata=False
            continue
        try:
            boro = tuple( i['boronic'])[0]
            borodata=boroEnc[boro]
        except:
            borodata=False
            continue
        #print(halodata, "\n",borodata)
        print( '\t'.join(halodata), '\t'.join(borodata), baseClass, solvClass, sep='\t' )
        if not base in allBases:
            allBases[base]=0
        if not solv in allSolvent:
            allSolvent[solv]=0
        allBases[base]+=1
        allSolvent[solv]+=1
    #for i in sorted(allBases, key=lambda x:allBases[x]):
    #    print(i, allBases[i])
    #print("S", allSolvent)
    #print("========")
    #for i in sorted(allSolvent, key=lambda x:allSolvent[x]):
    #    print(i, allSolvent[i])
