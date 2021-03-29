import numpy, sys
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
        data.append( {'solvent':ev[5], 'base':ev[6], 'temp':ev[7], 'yield':ev[11], 'boronic':boro, 'halogen':halo, 'ligand':ev[8] } )
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
    if s in ('ethylene glycol'):
        return 'EG'
    if s in ('acetonitrile', 'acetone'):
        return 'ace'
    if s in ('dichloromethane', ):
        return 'DCM'
    return 'other'


def makeOutput(base, solvent, oldSolvents=False):
    baseDict = { frozenset({'acetate'}):6, frozenset({'amine'}):5, frozenset({'hydroxide'}):4, frozenset({'fluoride'}):3,
                frozenset({'phosphate'}):2, frozenset({'carbonate'}):1, 
    }
    
    solventDictOld = {  frozenset({'alcohol'}):12, frozenset({'ace', 'water'}):11, frozenset({'water', 'aromatic'}):10,  frozenset({'water', 'alcohol'}):9,
                frozenset({'water'}):8, frozenset({'amide'}):7,  frozenset({'alcohol', 'aromatic'}):6, frozenset({'aromatic'}):5,  frozenset({'water', 'amide'}):4,
                frozenset({'water', 'alcohol', 'aromatic'}):3,   frozenset({'etheric'}):2,  frozenset({'water', 'etheric'}):1,
    }
    solventDict = { 'polar':1, 'aromatic':2, 'etheric':3, 'polarAromatic':4, 'waterEther':5, 'other':6}
    if oldSolvents:
        return baseDict.get(base, len(baseDict)+1), solventDictOld.get(solvent, len(solventDictOld)+1)
    return baseDict.get(base, len(baseDict)+1), solventDict.get(solvent, len(solventDict)+1)


def getRdkitDesc(smi):
    mol = Chem.MolFromSmiles(smi)
    return [ str(x[1](mol)) for x in Descriptors.descList]


def getIncoFP(smi, incoGr):
    mol=Chem.MolFromSmiles(smi)
    return [ str(sum([mol.HasSubstructMatch(gr) for gr in x ])) for x in incoGr]

def getLigandData(name, normalize):
    stdName={'Tedicyp': "tedicyp",
        'catacxium A':"CataCXium A",
        'xantphos':'Xantphos',
        'PPh3 DVB':'PPh3', 'PPh3, “Pd-cycle”': 'PPh3', 
        'P(o-tol)3': 'P(o-Tol)3',
        'P(tBu)3 DVB': 'PtBu3', 'P(tBu)3':'PtBu3', 
        'PCy3':"P(cychex)3",
        'APhos': 'AmPhos',
        'TPPTPS':'TPPTS', 'TPPS':'TPPTS',
    }

    ligandDB={ #{'type':'P', 'dentate':, 'cone':, '31P':, 'burn': },
       "Xantphos": {'type':'P', 'dentate':2, 'cone':162,               '31P':-18.1,            'bite':104.6}, 
         "dtbpf" : {'type':'P', 'dentate':2, 'cone':167,               '31P':24.6,             'bite':104},
        "XPhos"  : {'type':'P', 'dentate':1, 'cone':256,  'coneS':194, '31P':-12.3, 'burn':48.8 },
    "CataCXium A": {'type':'P', 'dentate':1, 'cone':176,  'coneS':161, '31P':24.9,  'burn':36.3 },
        "AmPhos" : {'type':'P', 'dentate':1, 'cone':170,               '31P':17.2,  },
     "P(cychex)3": {'type':'P', 'dentate':1, 'cone':170,  'coneS':139, '31P':9.8,   'burn':33.6 },
     "P(o-Tol)3" : {'type':'P', 'dentate':1, 'cone':194,  'coneS':153, '31P':-29.6, 'burn':37.3 },
        "PtBu3"  : {'type':'P', 'dentate':1, 'cone':182,  'coneS':141, '31P':63,    'burn':37.3 },
    #253 "TPPTS"  : {'type':'P', 'dentate':, 'cone':, '31P':, 'burn': },
        "SPhos"  : {'type':'P', 'dentate':1, 'cone':240,               '31P':53.7,  'burn':53.7 },
        "dppf"   : {'type':'P', 'dentate':2, 'cone':145,               '31P':-16.8,             'bite':98.74},
        "PPh3"   : {'type':'P', 'dentate':1, 'cone':145,  'coneS':141, '31P':-6,    'burn':30.9},
        "TPPTS"  : {'type':'P', 'dentate':1, 'cone':166,               '31P':-4.2 },
        "ruphos" : {'type':'P', 'dentate':1, 'cone':187.5,             '31P':-8.5 },
        "tedicyp": {'type':'P', 'dentate':4,                           '31P':-17.0 },
       "EvanPhos": {'type':'P', 'dentate':1,                           '31P':-9.2 },
    "CyJohnPhos" : {'type':'P', 'dentate':1, },
        "dppe"   : {'type':'P', 'dentate':2, },
    "trifuran-2-yl-phosphane" : {'type':'P', 'dentate':1, },
        "WePhos" : {'type':'P', 'dentate':1, },
    "HandaPhos"  : {'type':'P', 'dentate':1},
      'johnphos' : {'type':'P', 'dentate':1},

    }
    name=name.strip()
    name = stdName.get(name,name)
    if name in ('', 'no ligand'):
        #return False
        return [0,0, 0,0,0]
    if name in ligandDB:
        ligProp=ligandDB[name]
        # hasLigans, hasPhosphine, Ndentate, cone, 31P
        cone = ligProp.get('cone',0)
        pnmr =  ligProp.get('31P', 0)
        if normalize == 'scalled':
            if cone:
                cone = (cone /100)-2
            if pnmr:
                pnmr = (pnmr-15)/50
        return [1, 1, ligProp['dentate'], cone, pnmr ]
    print("IGNOREDNAME", name)
    #raise
    return False



class GetRepr():
    def __init__(self, repType, fplen=512):
        self.repType=repType
        self.fplen=fplen
        self.data=dict()
    def calcRep(self,smi):
        if self.repType == 'morgan3':
            self.data[smi] = AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(smi),3, nBits=self.fplen ).ToBitString()
        elif self.repType == 'canonSmiles':
            self.data[smi] = Chem.CanonSmiles(smi)
        elif self.repType == 'rdkit':
            self.data[smi] =  getRdkitDesc(smi)
        elif self.repType == 'morgan3rdkit':
            self.data[smi] = [x for x in AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(smi),3, nBits=self.fplen ).ToBitString()] + getRdkitDesc(smi)
        else:
            raise
    def getRep(self, smi):
        smi=Chem.CanonSmiles(smi)
        if not smi in self.data:
            self.calcRep(smi)
        return  self.data[smi]
    """
    if mode == 'enc':
        haloEnc = {x.split('\t')[0]:x.strip().split('\t')[1:] for x in open('encodedHalogens.8d') }
        boroEnc = {x.split('\t')[0]:x.strip().split('\t')[1:] for x in open('encodedBoronicAcid.8d') }
    elif mode == 'plain':
        haloEnc={smi.strip():desc.strip().split('\t') for smi, desc in zip(open('opisSubstratow/halogenicSmiles').readlines() , open('opisSubstratow/halogens').readlines()) }
        boroEnc={smi.strip():desc.strip().split('\t') for smi, desc in zip(open('opisSubstratow/boronicAcidSmiles').readlines() , open('opisSubstratow/boronociAcidsInp').readlines()) }
    elif mode == 'morgan3':
        fplen=512
        haloEnc = {smi.strip():[x for x in AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(smi),3,nBits=fplen ).ToBitString()] for smi in open('opisSubstratow/halogenicSmiles').readlines()  }
        boroEnc = {smi.strip():[x for x in ] for smi in open('opisSubstratow/boronicAcidSmiles').readlines()  }
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
    """

class EmbededConditions:
    def __init__(self, bases, solvents, ligands):
        self.numBases = bases
        self.numSolvents = solvents
        self.numLigands = ligands
        self.ligands = dict()
        self.solvents = dict()
        self.bases = dict()

    def getBases(self, inplist, returnType='str', sep='\t'):
        return self.getRep(inplist, self.bases, self.numBases, returnType, sep)
    def getSolvents(self, inplist, returnType='str', sep='\t'):
        return self.getRep(inplist, self.solvents, self.numSolvents, returnType, sep)
    def getLigands(self, inplist, returnType='str', sep='\t'):
        return self.getRep(inplist, self.ligands, self.numLigands, returnType, sep )

    def getRep(self, inplist, dictionary, length, returnType, sep):
        retlist = [ 0 for x in range(length)]
        for pos, b in enumerate(inplist):
            if not b in dictionary:
                dictionary[b] = len(dictionary)+1
            try:
                retlist[pos] = dictionary[b]
            except:
                print("==>", inplist, length, "ignore", b )
                continue
                #raise
        if returnType == 'str':
            return sep.join([str(x) for x in retlist])
        return retlist
    def stat(self):
        print("ligands", len(self.ligands), "bases", len(self.bases), "solvent", len(self.solvents), file=sys.stderr)


def getMclassPairs(allRx, minDiff):
    #allReactions[rxSubstrates][temp][rxConditions].append( (halodata, borodata, rxYield) )
    #Sclass = []
    Mclass = []
    for rxsbs in allRx:
        for temp in allRx[rxsbs]:
            status = 'NONE'
            dane=[]
            yields=[]
            pos = []
            if len( allRx[rxsbs][temp]) == 1: #only one condition
                status = "Sclass"
            for cond in allRx[rxsbs][temp]:
                if len(allRx[rxsbs][temp][cond]) > 1: #more that one reaction in same conditions ignore it
                    status = 'Sclass'
                for rxYield, rxPos in allRx[rxsbs][temp][cond]:
                    #halo, boro, ligandData, rxYield = rxinfo
                    yields.append(rxYield)
                    pos.append(rxPos)
            if status == 'NONE' and len(yields) >= 2 and abs( max(yields)-min(yields) )> minDiff:
                status = 'Mclass' #+str( len(i))+'_'+str(yields)
                maxidx = [ i for i,x in enumerate(yields) if x == max(yields) ]
                minidx = [ i for i,x in enumerate(yields) if x == min(yields) ]
                if len(maxidx) < 1 or len(minidx) < 1:
                    print("MAX", maxidx, "MIN", minidx)
                    raise
                maxidx = maxidx[0]
                minidx = minidx[0]
                if maxidx == minidx:
                    raise
                Mclass.extend( [pos[maxidx], pos[minidx]])
    return Mclass



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['morgan3', 'enc', 'canonSmiles', 'plain', 'forGCNN', 'rdkit', 'morgan3rdkit'], default='morgan3')
    parser.add_argument('--conditions', choices=['newClasses', 'oldClasses', 'embedded', 'newClassesEmbedLig', 'oldClassesEmbedLig'], default='newClasses')
    parser.add_argument('--includeligand', choices=['raw', 'scalled'])
    parser.add_argument('--allownotemp', action='store_true')
    parser.add_argument('--userawtemperature', action='store_true')
    parser.add_argument('--outputMclass', type=str)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    print("ARGS", args)
    lines=[]
    #mode='rdkit' #'enc'
    #mode = 'rdkit+morgan'
    #mode = 'plain' #+rdkit+morgan'
    #mode='morgan3' #+inco'
    #mode='inputForGCNN'
    sbsRep = GetRepr(args.mode)
    embCond = EmbededConditions(bases=2, solvents=4, ligands=2)
    #for fn in sys.argv[1:]:
    for line in open(args.input):
        lines.append(line)
    dane = parse(lines)
    allBases=dict()
    allSolvent=dict()
    outfile = open(args.output, 'w')
    allrxes = dict()
    rxPos = 0
    allLinesToPrint = []
    for i in dane:
        if len(i['boronic']) != 1 or len(i['halogen']) !=1:
            print("IGNORED too many sbs", len(i['boronic']), len(i['halogen']), i)
            continue
        base = frozenset([getBaseClass(b) for b in i['base'] ]) 
        #if args.useoldsolvent:
        #    solv =getSolventClass(  frozenset([ getSolventClass(s) for s in i['solvent'] ]) )
        #else:
        if args.conditions == 'oldClasses' or args.conditions == 'oldClassesEmbedLig':
            solv = frozenset([ getSolventClassAP(s) for s in i['solvent'] ])
            baseClass, solvClass = makeOutput(base ,solv, oldSolvents=True)
        elif args.conditions == 'newClasses' or args.conditions == 'newClassesEmbedLig':
            solv =getSolventClass(  frozenset([ getSolventClassAP(s) for s in i['solvent'] ]) )
            baseClass, solvClass = makeOutput(base ,solv, oldSolvents=False)
        
        if args.conditions == 'embedded':
            baseClass = embCond.getBases(i['base'])
            solvClass = embCond.getSolvents( i['solvent'])
            ligandData = embCond.getLigands( i['ligand'])
        elif args.conditions == 'newClassesEmbedLig' or args.conditions == 'oldClassesEmbedLig':
            ligandData = embCond.getLigands( i['ligand'])
        else:
            ligandData=[ getLigandData(l, args.includeligand) for l in i['ligand'] ]
            ligandData = [ x for x in ligandData if x]
            if not ligandData:
                ligandData =[ getLigandData('', args.includeligand), ]
            if len(ligandData) > 1:
                avgVal = [ sum([l[pos] for l in ligandData ] )/len(ligandData) for pos in range( len(ligandData[0]) ) ]
                #print("ZZZZ", i['ligand'], ligandData, "==>", avgVal)
                ligandData = avgVal
            else: 
                ligandData = ligandData[0]
        #print("SOL", i['solvent'], solvClass, args, solvClass)
        rxYield=max( i['yield'])
        try:
            halo = tuple(i['halogen'])[0]
            #halodata= haloEnc[halo]
            halodata = sbsRep.getRep(halo)
        except:
            halodata=False
            print("IGNORED2", i)
            #continue
            raise
        try:
            boro = tuple( i['boronic'])[0]
            #borodata=boroEnc[boro]
            borodata = sbsRep.getRep(boro)
        except:
            borodata=False
            print("IGNORED3", i)
            continue
        #print(halodata, "\n",borodata)
        if not ligandData:
            print( 'IGNORED, ligandData', i['ligand'])
            continue #ignore strange ligand
        if not i['temp']:
            if args.allownotemp:
                i['temp']=[0,]
            else:
                print("ignored 4", i)
                continue
        temp= max(i['temp'])
        if not args.userawtemperature:
            temp = (temp-90)/100
        if args.mode == 'canonSmiles':
            borons = borodata
            halogens = halodata
        else:
            borons = '\t'.join(borodata)
            halogens = '\t'.join(halodata)
        if args.conditions == 'embedded' or args.conditions == 'newClassesEmbedLig' or args.conditions == 'oldClassesEmbedLig':
            listToPrint = [baseClass, solvent, ligandData, temp, halogens, borons, rxYield]
        elif args.includeligand:
            listToPrint = [baseClass, solvClass, halogens, borons, temp, '\t'.join([str(x) for x in ligandData]), rxYield]
        else:
            listToPrint = [baseClass, solvClass, halogens, borons, temp, rxYield]
        lineToPrint =  '\t'.join([str(x) for x in listToPrint])
        if args.outputMclass:
            sbs = (halogens, borons)
            if not sbs in allrxes:
                allrxes[sbs]= dict()
            if not temp in allrxes[sbs]:
                allrxes[sbs][temp]= dict()
            cond = (baseClass, solvClass)
            #if args.includeligand:
            #    cond = (baseClass, solvClass, ligandData)
            if not cond in allrxes[sbs][temp]:
                allrxes[sbs][temp][cond]=[]
            allrxes[sbs][temp][cond].append( (rxYield, rxPos) )
            rxPos += 1
            allLinesToPrint.append(lineToPrint)
        else:
            print(lineToPrint, file=outfile)
    if args.outputMclass:
        outMfile = open(args.outputMclass, 'w')
        mclass = getMclassPairs(allrxes, minDiff=25)
        mclasset = set(mclass)
        for rxNum in range( rxPos):
            if rxNum in mclasset:
                continue
            print(allLinesToPrint[rxNum], file=outfile)
            #print(rxNum, len(lineToPrint), rxPos)
        for midx in mclass:
            print(allLinesToPrint[midx], file=outMfile)
        #print("MC", len(mclass), mclass )
        outMfile.close()
    outfile.close()
    if args.conditions == 'embedded':
        embCond.stat()