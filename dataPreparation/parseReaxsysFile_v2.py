import itertools, re
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def parseArgs():
    import argparse
    parser = argparse.ArgumentParser(description='parser of Suzuki reaction in csv from reaxsys')
    parser.add_argument('--heterohetero', action="store_true", default=False, help='include heteroaryl heteroaryl Suzuki coupling data')
    parser.add_argument('--arylhetero', action="store_true", default=False, help='include aryl heteroaryl Suzuki coupling data')
    parser.add_argument('--arylaryl', action="store_true", default=False, help='include aryl aryl Suzuki coupling data')
    parser.add_argument('--withpatents', action='store_true', default=False, help='include also data from patents')
    parser.add_argument('--withtemponly', action='store_true', default=False, help='include only reaction with given temperature')
    parser.add_argument('--withbaseonly', action='store_true', default=False, help='include only reaction with given base')
    parser.add_argument('--withsolvonly', action='store_true', default=False, help='include only reaction with given solvent')
    parser.add_argument('--withyieldonly', action='store_true', default=False, help='include only reaction with given yield')
    parser.add_argument('--withligandonly', action='store_true', default=False, help='include only reaction with given ligand')
    parser.add_argument('--withpdonly', action='store_true', default=False, help='include only reaction with given Pd-source')
    parser.add_argument('--output', default='plikzestatamisuzukiego_5', help='output file name')
    args = parser.parse_args()
    if args.heterohetero is False and args.arylhetero is False and args.arylaryl == False:
       parser.error("at least one of --heterohetero and --arylhetero required")
    return args


def parseFile(fn, separator='\t', includePatents=True):
    lines =open(fn).readlines()
    #HEAD ['Reaction ID', 'Reaction: Links to Reaxys', 'Data Count', 'Number of Reaction Details', 'Reaction Rank', 'Record Type', 'Reactant', 'Product', 'Bin', 'Reaction', 
    #10: 'Reaction Details: Reaction Classification', 'Example label', 'Example title', 'Fulltext of reaction', 'Number of Reaction Steps', 'Multi-step Scheme', 
    #15: 'Multi-step Details', 'Number of Stages', 'Solid Phase', 'Time (Reaction Details) [h]', 'Temperature (Reaction Details) [C]', 
    #20: 'Pressure (Reaction Details) [Torr]', 'pH-Value (Reaction Details)', 'Other Conditions', 'Reaction Type', 'Subject Studied', 
    #25: 'Prototype Reaction', 'Named Reaction', 'Type of reaction description (Reaction Details)', 'Location', 'Comment (Reaction Details)', 'Product', 
    #30: 'Yield', 'Yield (numerical)', 'Yield (optical)', 'Stage Reactant', 'Reagent', 'Catalyst', 'Solvent (Reaction Details)', 'References', 'Links to Reaxys']
    head= [x.strip() for x in lines[0].split(separator) if x.strip()]
    if includePatents:
        data= [ line.split(separator) for line in lines[1:] ]
    else:
        data= [ line.split(separator) for line in lines[1:] if not 'Patent;' in line ]
    return {'header':head, 'data':data}


def _getIdxOfAromNeiOfAtm(atomObj):
    neis= atomObj.GetNeighbors()
    if len(neis) == 1:
        return neis[0].GetIdx()
    arom = [n for n in neis if n.GetIsAromatic()]
    if len(arom) == 0:
        arom = [ n for n in neis if str(n.GetHybridization()) == 'SP2' ]
    if len(arom) != 1: 
        print("XXX", arom, "ATOM", atomObj, Chem.MolToSmiles(atomObj.GetOwningMol()) )
        raise
    return arom[0].GetIdx()

def _getRingSystem(idx, inRingIdxes):
    
    ri = ()
    added=set()
    front=[]
    ##1st ring
    for ring in inRingIdxes:
        if idx in ring:
            ri = ring
            added.add(ring)
            front = ring
            break
    ##2nd rings
    front2=[]
    ri2=[]
    for ring in inRingIdxes:
        if ring in added: continue
        if any([idx in front for idx in ring]):
            front2.extend( ring)
            added.add(ring)
            ri2.append(ring)
    if not ri2:
        return [ ri, ]
    #3rd ring
    ri3=[]
    for ring in inRingIdxes:
        if ring in added: continue
        if any([idx in front2 for idx in ring]):
            added.add(ring)
            ri3.append(ring)
    if not ri3:
        return [ri, ri2]
    return [ri, ri2, ri3]


def getRingBenzo(mol, expectedAtoms):
    allTypes={'benzoin6m1N':{'c1ccc2ncccc2c1':'', 'c1ccc2cnccc2c1':'',},
        'benzoIn6m2N':{'c1ccc2ncncc2c1':'', 'c1ccc2cnncc2c1':'', 'c1ccc2nccnc2c1':'', 'c1ccc2nnccc2c1':'',},
        'benzoIn6m3N':{'c1ccc2nnncc2c1':'', 'c1ccc2nncnc2c1':'',},
        'benzoIn6m4N':{'c1ccc2nnnnc2c1':''},
        'benzoIn5m1het':{'c1cccc2c1[nX3]cc2':'', 'c1cccc2c1occ2':'', 'c1cccc2c1scc2':'', 'c1cccc2c1[se]cc2':'', 'c1cccc2c1c[nX3]c2':'', 'c1cccc2c1coc2':'', 'c1cccc2c1csc2':'', 'c1cccc2c1c[se]c2':''},
        'benzoIn5m2het':{'c1cccc2c1noc2':'', 'c1cccc2c1nsc2':'', 'c1cccc2c1nnc2':'', 'c1cccc2c1nco2':'', 'c1cccc2c1ncs2':'', 'c1cccc2c1ncn2':'',  
            'c1cccc2c1onc2':'', 'c1cccc2c1snc2':'', 'c1cccc2c1[se]cn2':'', 'c1cccc2c1[se]nc2':'',},
        'benzoIn5m3het':{'c1cccc2c1nnn2':'', 'c1cccc2c1non2':'', 'c1cccc2c1onn2':'', 'c1cccc2c1nsn2':'', 'c1cccc2c1snn2':'', 'c1cccc2c1[se]nn2':'', 'c1cccc2c1n[se]n2':''},
    }
    
    for rclass in allTypes:
        for rsmarts in allTypes[rclass]:
            smamol = Chem.MolFromSmarts( rsmarts)
            if not smamol:
                print("CANNOT PARSE", smamol, rsmarts)
            allTypes[rclass][rsmarts] = smamol

    foundRings={}
    for rclass in allTypes:
        for rsmarts in allTypes[rclass]:
            allFound =  mol.GetSubstructMatches( allTypes[rclass][rsmarts] )
            if allFound:
                #print("RIMOL", mol, allTypes[rclass][rsmarts], "ALLFOUND",  allFound,)
                found =[ring for ring in  allFound if any(expAtom in ring for expAtom in expectedAtoms)]
                if found:
                    foundRings[rclass+":::"+rsmarts]= list(found)
    submols=dict()
    for fRingType in foundRings:
        numAtomInMol=mol.GetNumAtoms()
        fRingClass = fRingType.split(':::')[0]
        for fRingIdx in foundRings[fRingType]:
            editMol = Chem.EditableMol(mol)
            allIdxes= sorted(fRingIdx)
            for atm in expectedAtoms:
                if atm in fRingIdx:
                    allIdxes.append( expectedAtoms[atm] )
            allIdxes=set(allIdxes)
            for idx in range(numAtomInMol-1, -1, -1):
                if not idx in allIdxes:
                    editMol.RemoveAtom(idx)
            submols[ fRingClass+':::'+Chem.MolToSmiles( editMol.GetMol() ) ]=allIdxes
    return submols
    #return foundRings




def getRingNames(mol, which, neiOfWhich ):
    """
        mol - mol Object
        which - Br or B
        neiOfWhich - dict {neiOfWhichIdx:idxOfWhich, }
    """
    #mm=Chem.MolFromSmarts('c1cccc2c1nnc2')
    #if (mol.HasSubstructMatch(mm)):print("DZIALA") 
    allTypesB={
        'FURANS AND BENZOFURANS': { 'OB(O)c1ccco1':'2-iodo-furan', 'OB(O)c1cc2aaaac2o1':'2-iodobenzofuran', 'c1c(B(O)O)cco1':'3-iodobenzofuran', 'c1c(B(O)O)c2:a:a:a:a:c2o1':'3-iodo-furan', },
        'THIOPHENE AND BENZOTHIOPHENE': {'OB(O)c1cccs1':'2-iodo-thiazole', 'OB(O)c1cc2aaaac2s1':'2-iodobenzothiazole', 'c1c(B(O)O)ccs1':'3-iodo-thiazole', 'c1c(B(O)O)c2:a:a:a:a:c2s1':'3-iodobenzothiazole'},
        'PYRROLE AND INDOLE': {'OB(O)c1cccn1': '', 'OB(O)c1cc2aaaac2n1':'', 'c1c(B(O)O)ccn1':'','c1c(B(O)O)c2:a:a:a:a:c2n1':''},
        'ISOOXAZOLE': {'OB(O)c1ccno1':'', 'OB(O)c1onc2:a:a:a:a:c12':'', 'c1c(B(O)O)cno1':'', 'c1cc(B(O)O)no1':'', 'OB(O)c1noc2:a:a:a:a:c12':''},
        'OXAZOLE AND BENZOXAZOLE': { 'OB(O)c1cnco1':'', 'OB(O)c1cocn1':'', 'OB(O)c1ncco1':'', 'OB(O)c1nc2aaaac2o1':''},
        'THIAZOLE AND BENZOTHIAZOLE': { 'OB(O)c1cncs1':'', 'OB(O)c1cscn1':'', 'OB(O)c1nccs1':'', 'OB(O)c1nc2aaaac2s1':'', },
        'ISOTHIAZOLE AND ISOBENZOTHIAZOLE':{'OB(O)c1ccns1':'', 'OB(O)c1snc2:a:a:a:a:c12':'', 'c1c(B(O)O)cns1':'', 'c1cc(B(O)O)ns1':'', 'OB(O)c1nsc2:a:a:a:a:c12':'',},
        'PYRAZOLE AND BENZOPYRAZOLE': { 'OB(O)c1nncc1':'', 'OB(O)c1nnc2aaaac12':'', 'c1c(B(O)O)cnn1':'', 'OB(O)c1cnn2aaaac12':'', 'OB(O)c1cc2aaaan2n1':'', 'OB(O)c1ccnn1':'','OB(O)c1c2aaaac2nn1':'', },
        'IMIDAZOLE AND BENZIMIDAZOLE': { 'OB(O)c1cncn1':'', 'OB(O)c1c2aaaan2cn1':'', 'OB(O)c1cn2aaaac2n1':'', 'OB(O)c1cncn1':'', 'OB(O)c1cnc2aaaan12':'', 'OB(O)c1nccn1':'','OB(O)c1nc2aaaac2n1':'', 'OB(O)c1ncc2aaaan12':'',},
        '1,2,5-OXADIAZOLE': { 'n1oncc1B(O)O':''},
        '1,2,4-OXADIAZOLE': {'OB(O)c1ncno1':'', 'OB(O)c1ncon1':''},
        '1,3,4-OXADIAZOLE': {'n1ncoc1B(O)O':'',  },
        '1,2,5-THIADIAZOLE': {'n1sncc1B(O)O':''},
        '1,2,4-THIADIAZOLE':{'OB(O)c1ncns1':'', 'OB(O)c1ncsn1':''},
        '1,3,4-THIADIAZOLE':{'n1ncsc1B(O)O':'' , },
        '1H-1,2,3-TRIAZOLE':{'OB(O)c1cnnn1':'', 'c1c(B(O)O)nnn1':'', 'OB(O)c1c2a=aa=an2nn1':'', },
        '2H-1,2,3-TRIAZOLE':{'OB(O)c1nnnc1':'',},
        '1H-1,2,4-TRIAZOLE':{'OB(O)c1ncnn1':'', 'c1nc(B(O)O)nn1':'', 'OB(O)c1nn2aaaac2n1':'', },
        '4H-1,2,4-TRIAZOLE':{'OB(O)c1nncn1':'', 'OB(O)c1nnc2aaaan12':'', },
        'TETRAZOLE': { 'OB(O)c1nnnn1':'', 'OB(O)c1nnnn1':'', },

        'PYRIDINES': {'OB(O)c1ccccn1':'6mn','OB(O)c1cnccc1':'3-pyridine','OB(O)c1ccncc1':'4-pyridine'},
        'PYRIDAZINE':{'OB(O)c1cccnn1':'6mnn','OB(O)c1cnncc1':'4-pyridazine',},
        'PYRIMIDINE':{'OB(O)c1ncccn1':'2-iodopyrimidine','OB(O)c1ccncn1':'4-iodopyrimidine', 'OB(O)c1cncnc1':'5-iodopyrimidine'},
        'PYRAZINE':{'OB(O)c1cnccn1':'2-iodopyrazine',},
        '1,2,3-triazine': {'OB(O)c1ccnnn1':'4-iodo-1,2,3-triazine', 'OB(O)c1cnnnc1':'5-iodo-1,2,3-triazine',},
        '1,2,4-triazine':{'OB(O)c1nnccn1':'3-iodo-1,2,4-triazine', 'OB(O)c1nncnc1':'6-iodo-1,2,4-triazine', 'OB(O)c1ncnnc1':'5-iodo-1,2,4-triazine'},
        '1,3,5-triazine': {'OB(O)c1ncncn1':'2-iodo-1,3,5-triazine', },
        '6-membrede with 4-heteroatoms': {'OB(O)c1nncnn1':'3-iodo-1,2,4,5-tetrazine'},
        '5-membrede with selenide': {'OB(O)c1ccc[Se]1':''},
    }
    allTypesBr={
        'FURANS AND BENZOFURANS': { 'Ic1ccco1':'2-iodo-furan', 'Ic1cc2aaaac2o1':'2-iodobenzofuran', 'c1c(I)cco1':'3-iodobenzofuran', 'c1c(I)c2:a:a:a:a:c2o1':'3-iodo-furan', 
            'Brc1ccco1':'2-bromo-furan', 'Brc1cc2aaaac2o1': '2-bromobenzofuran', 'c1c(Br)cco1':'3-bromo-furan', 'c1c(Br)c2:a:a:a:a:c2o1':'3-bromobenzofuran',
            'Clc1ccco1':'2-bromo-furan', 'Clc1cc2aaaac2o1': '2-bromobenzofuran', 'c1c(Cl)cco1':'3-bromo-furan', 'c1c(Cl)c2:a:a:a:a:c2o1':'3-bromobenzofuran'},
        'THIOPHENE AND BENZOTHIOPHENE': {'Ic1cccs1':'2-iodo-thiazole', 'Ic1cc2aaaac2s1':'2-iodobenzothiazole', 'c1c(I)c=cs1':'3-iodo-thiazole','c1c(I)c2:a:a:a:a:c2s1':'3-iodobenzothiazole',
            'Brc1cccs1':'2-bromo-thiazole', 'Brc1cc2aaaac2s1':'2-bromobenzothiazole', 'c1c(Br)ccs1':'3-bromo-thiazole', 'c1c(Br)c2:a:a:a:a:c2s1':'3-bromobenzothiazole',
            'Clc1cccs1':'2-bromo-thiazole', 'Clc1cc2aaaac2s1':'2-bromobenzothiazole', 'c1c(Cl)ccs1':'3-bromo-thiazole', 'c1c(Cl)c2:a:a:a:a:c2s1':'3-bromobenzothiazole',},
        'PYRROLE AND INDOLE': {'Ic1cccn1': '', 'Ic1cc2aaaac2n1':'', 'c1c(I)ccn1':'','c1c(I)c2:a:a:a:a:c2n1':'', 'Brc1cccn1':'','Brc1cc2aaaac2n1':'', 'c1c(Br)ccn1':'', 'c1c(Br)c2:a:a:a:a:c2n1':'',
            'Clc1cccn1':'','Clc1cc2aaaac2n1':'', 'c1c(Cl)ccn1':'', 'c1c(Cl)c2:a:a:a:a:c2n1':''},
        'ISOOXAZOLE': {'Ic1ccno1':'', 'Ic1onc2:a:a:a:a:c12':'', 'c1c(I)cno1':'', 'c1cc(I)no1':'', 'Ic1noc2:a:a:a:a:c12':'', 'Brc1ccno1':'','Brc1onc2:a:a:a:a:c12':'', 'c1c(Br)cno1':'', 'c1cc(Br)no1':'', 'Brc1noc2:a:a:a:a:c12':'',
            'Clc1ccno1':'','Clc1onc2:a:a:a:a:c12':'', 'c1c(Cl)cno1':'', 'c1cc(Cl)no1':'', 'Clc1noc2:a:a:a:a:c12':''},
        'OXAZOLE AND BENZOXAZOLE': { 'Ic1cnco1':'', 'Ic1cocn1':'', 'Ic1ncco1':'', 'Ic1nc2aaaac2o1':'', 'Brc1cnco1':'', 'Brc1cocn1':'', 'Brc1ncco1':'', 'Brc1nc2aaaac2o1':'',
             'Clc1cnco1':'', 'Clc1cocn1':'', 'Clc1ncco1':'', 'Clc1nc2aaaac2o1':'',},
        'THIAZOLE AND BENZOTHIAZOLE': { 'Ic1cncs1':'', 'Ic1cscn1':'', 'Ic1nccs1':'', 'Ic1nc2aaaac2s1':'', 'Brc1cncs1':'', 'Brc1cscn1':'', 'Brc1nccs1':'', 'Brc1nc2aaaac2s1':'',
            'Clc1cncs1':'', 'Clc1cscn1':'', 'Clc1nccs1':'', 'Clc1nc2aaaac2s1':'',},
        'ISOTHIAZOLE AND ISOBENZOTHIAZOLE':{'Ic1ccns1':'', 'Ic1snc2:a:a:a:a:c12':'', 'c1c(I)cns1':'', 'c1cc(I)ns1':'', 'Ic1nsc2:a:a:a:a:c12':'',
            'Brc1ccns1':'', 'Brc1snc2:a:a:a:a:c12':'', 'c1c(Br)cns1':'', 'c1cc(Br)ns1':'', 'Brc1nsc2:a:a:a:a:c12':'',
            'Clc1ccns1':'', 'Clc1snc2:a:a:a:a:c12':'', 'c1c(Cl)cns1':'', 'c1cc(Cl)ns1':'', 'Clc1nsc2:a:a:a:a:c12':'',},
        'PYRAZOLE AND BENZOPYRAZOLE': { 'Ic1nncc1':'', 'Ic1nnc2aaaac12':'', 'c1c(I)cnn1':'', 'Ic1cnn2aaaac12':'', 'Ic1cc2aaaan2n1':'', 'Ic1ccnn1':'','Ic1c2aaaac2nn1':'', 
            'BrC1=NNC=C1':'', 'BrC1=NNc2aaaac12':'', 'c1c(Br)cnn1':'', 'Brc1cnn2aaaac12':'', 'Brc1cc2aaaan2n1':'', 'Brc1ccnn1':'', 'Brc1c2aaaac2nn1':'', 
            'ClC1=NNC=C1':'', 'ClC1=NNc2aaaac12':'', 'c1c(Cl)cnn1':'', 'Clc1cnn2aaaac12':'', 'Clc1cc2aaaan2n1':'', 'Clc1ccnn1':'', 'Clc1c2aaaac2nn1':'', },
        'IMIDAZOLE AND BENZIMIDAZOLE': { 'Ic1cncn1':'', 'Ic1c2aaaan2cn1':'', 'Ic1cn2aaaac2n1':'', 'Ic1cncn1':'', 'Ic1cnc2aaaan12':'', 'Ic1nccn1':'','Ic1nc2aaaac2n1':'', 'Ic1ncc2aaaan12':'',
            'Brc1cncn1':'', 'Brc1c2aaaan2cn1':'', 'Brc1cn2aaaac2n1':'', 'Brc1cncn1':'', 'Brc1cnc2aaaan12':'','Brc1nccn1':'', 'Brc1nc2aaaac2n1':'', 'Brc1ncc2aaaan12':'',
            'Clc1cncn1':'', 'Clc1c2aaaan2cn1':'', 'Clc1cn2aaaac2n1':'', 'Clc1cncn1':'', 'Clc1cnc2aaaan12':'','Clc1nccn1':'', 'Clc1nc2aaaac2n1':'', 'Clc1ncc2aaaan12':'',},
        '1,2,5-OXADIAZOLE': { 'n1oncc1I':'', 'n1oncc1Br':'', 'n1oncc1Cl':'',},
        '1,2,4-OXADIAZOLE': {'Ic1ncno1':'', 'Ic1ncon1':'', 'Brc1ncno1':'', 'Brc1ncon1':'', 'Clc1ncno1':'', 'Clc1ncon1':'',},
        '1,3,4-OXADIAZOLE': {'n1ncoc1I':'', 'n1ncoc1Br':'',  'n1ncoc1Cl':'',},
        '1,2,5-THIADIAZOLE': {'n1sncc1I':'', 'n1sncc1Br':'', 'n1sncc1Cl':'',},
        '1,2,4-THIADIAZOLE':{'Ic1ncns1':'', 'Ic1ncsn1':'', 'Brc1ncns1':'', 'Brc1ncsn1':'', 'Clc1ncns1':'', 'Clc1ncsn1':'',},
        '1,3,4-THIADIAZOLE':{'n1ncsc1I':'', 'n1ncsc1Br':'',  'n1ncsc1Cl':'',},
        '1H-1,2,3-TRIAZOLE':{'Ic1cnnn1':'', 'c1c(I)nnn1':'', 'Ic1c2a=aa=an2nn1':'', 'Brc1cnnn1':'', 'c1c(Br)nnn1':'', 'Brc1c2aaaan2nn1':'', 'Clc1cnnn1':'', 'c1c(Cl)nnn1':'', 'Clc1c2aaaan2nn1':'',},
        '2H-1,2,3-TRIAZOLE':{'Ic1nnnc1':'','Brc1nnnc1':'', 'Clc1nnnc1':'',},
        '1H-1,2,4-TRIAZOLE':{'Ic1ncnn1':'', 'c1nc(I)nn1':'', 'Ic1nn2aaaac2n1':'', 'Brc1ncnn1':'', 'c1nc(Br)nn1':'', 'Brc1nn2aaaac2n1':'',  'Clc1ncnn1':'', 'c1nc(Cl)nn1':'', 'Clc1nn2aaaac2n1':'',},
        '4H-1,2,4-TRIAZOLE':{'Ic1nncn1':'', 'Ic1nnc2aaaan12':'', 'Brc1nncn1':'', 'Brc1nnc2aaaan12':'',  'Clc1nncn1':'', 'Clc1nnc2aaaan12':'',},
        'TETRAZOLE': { 'Ic1nnnn1':'', 'Ic1nnnn1':'', 'Brc1nnnn1':'', 'Brc1nnnn1':'', },
        '1,2,3,4-THIATRIAZOLE': {'Ic1nnns1':'', 'Brc1nnns1':'',},
        '1,2,3,4-OXATRIAZOLE':{'Ic1nnno1':'', 'Brc1nnno1':'',},
        'with selenide': { 'Brc1ccc[Se]1':'', 'IC1=CC=C[Se]1':'', 'BrC1=C[Se]C=C1':'', 'IC1=C[Se]C=C1':'', 'BrC1=NC=C[Se]1':'', 'IC1=NC=C[Se]1':'', 'BrC1=CN=C[Se]1':'',
            'IC1=CN=C[Se]1':'', 'BrC1=CC=N[Se]1':'', 'IC1=CC=N[Se]1':'', 'BrC1=C[Se]N=C1':'', 'IC1=C[Se]N=C1':'', 'BrC1=C[Se]C=N1':'', 'IC1=C[Se]C=N1':'', 'BrC1=N[Se]C=C1':'',
            'IC1=N[Se]C=C1':'',},
        'PYRIDINES': {'Ic1ccccn1':'2-iodopyridine', 'Brc1ccccn1':'2-bromopyridine', 'Ic1cnccc1':'3-iodopyridine', 'Brc1cnccc1':'3-bromopyridine','Ic1ccncc1':'4-iodopyridine', 'Brc1ccncc1':'4-bromopyridine',
            'Clc1ccccn1':'2-bromopyridine', 'Clc1cnccc1':'3-iodopyridine',  'Clc1ccncc1':'4-bromopyridine',},
        'PYRIDAZINE': {'Ic1cccnn1':'3-iodopyridazine', 'Brc1cccnn1':'3-bromopyridazine', 'Ic1cnncc1':'4-iodopyridazine', 'Brc1cnncc1':'4-bromopyridazine'},
        'PYRIMIDINE':{'Ic1ncccn1':'2-iodopyrimidine','Brc1ncccn1':'2-bromopyrimidine', 'Ic1ccncn1':'4-iodopyrimidine', 'Brc1ccncn1':'4-bromopyrimidine', 'Ic1cncnc1':'5-iodopyrimidine', 'Brc1cncnc1':'5-bromopyrimidine',},
        'PYRAZINE':{'Ic1cnccn1':'2-iodopyrazine', 'Brc1cnccn1':'2-bromopyrazine',},
        '1,2,3-triazine': {'Ic1ccnnn1':'4-iodo-1,2,3-triazine', 'Brc1ccnnn1':'4-bromo-1,2,3-triazine', 'Ic1cnnnc1':'5-iodo-1,2,3-triazine','Brc1cnnnc1':'5-bromo-1,2,3-triazine'},
        '1,2,4-triazine': {'Ic1nnccn1':'3-iodo-1,2,4-triazine', 'Brc1nnccn1':'3-bromo-1,2,4-triazine', 'Ic1cncnn1':'6-iodo-1,2,4-triazine', 'Brc1nncnc1':'6-bromo-1,2,4-triazine',
                'Ic1cnncn1':'5-iodo-1,2,4-triazine', 'Brc1cnncn1':'5-bromo-1,2,4-triazine',},
        '1,3,5-triazine': {'Ic1ncncn1':'2-iodo-1,3,5-triazine', 'Brc1ncncn1':'2-bromo-1,3,5-triazine',},
        '6-membrede with 4-heteroatoms': {'Ic1nncnn1':'3-iodo-1,2,4,5-tetrazine', 'Brc1nncnn1':'3-bromo-1,2,4,5-tetrazine'}

    }

    for rclass in allTypesBr:
        for rsmarts in allTypesBr[rclass]:
            allTypesBr[rclass][rsmarts] = Chem.MolFromSmarts( rsmarts)

    for rclass in allTypesB:
        for rsmarts in allTypesB[rclass]:
            allTypesB[rclass][rsmarts] = Chem.MolFromSmarts( rsmarts)



    ####
    if which == 'Br' or which=='I' or which=='Cl':
        allTypes = allTypesBr
    elif which =='B':
        allTypes = allTypesB

    foundRings={}
    for rclass in allTypes:
        for rsmarts in allTypes[rclass]:
            found = mol.GetSubstructMatches( allTypes[rclass][rsmarts] )
            if found:
                foundRings[rclass+':::'+rsmarts]= list(found)
    ###
    
    #reduce
    #RN {'Brc1nccn1': ((0, 1, 2, 3, 8, 9),), 'Brc1nc2aaaac2n1': ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    if len(foundRings) > 1: #doreduce
        ringLens=dict()
        for rn in foundRings:
            for ring in foundRings[rn]:
                size=len(ring)
                if not size in ringLens:
                    ringLens[size]=[]
                ringLens[size].append(ring)
        ringsizes= sorted( ringLens.keys(), reverse=True)
        ringToRemove=set()
        for rsizeBig in ringsizes:
            for oneRingBig in ringLens[ rsizeBig]:
                for rsize in ringLens:
                    if rsize >= rsizeBig: continue
                    for idxSmallRing in range( len(ringLens[rsize])-1, -1,-1):
                        if all([a in oneRingBig for a in ringLens[rsize][idxSmallRing]]):
                            rem= ringLens[rsize].pop(idxSmallRing)
                            ringToRemove.add( rem)
        for ringToRm in  ringToRemove:
            keyToRm=[]
            for rtype in foundRings:
                if ringToRm in foundRings[rtype]:
                    foundRings[rtype].remove(ringToRm)
                if not foundRings[rtype]:
                    keyToRm.append(rtype)
            for k in keyToRm:
                foundRings.pop(k)
    if not foundRings:
        benzoRing=getRingBenzo(mol, neiOfWhich)
        #print("BENZO", benzoRing, Chem.MolToSmiles(mol), "N", neiOfWhich)
        return benzoRing
    return foundRings





def getRingType(smi, wantedAtmType='Br' ):
    #thiophene, furan, benzothiophene, benzofuran, pyrrole, pyrazole, thiazole, quinoline, isoquinoline, pyridine, triazole, benzooxadiazole)
    
    mol = Chem.MolFromSmiles(smi)
    atoms = [a for a in mol.GetAtoms()]
    symbols = [ a.GetSymbol() for a in atoms]
    idxOfWantedAtmType = [ i for i,s in enumerate(symbols) if s == wantedAtmType]
    #print("SMI", smi, "WANTED", wantedAtmType)
    idxOfNeiOfWantedAtmType = dict()
    toRm=[]
    for idx in idxOfWantedAtmType:
        #{ _getIdxOfAromNeiOfAtm(atoms[i] ):i for i in idxOfWantedAtmType}
        try:
            idxOfAromNei = _getIdxOfAromNeiOfAtm(atoms[idx] )
            idxOfNeiOfWantedAtmType[ idxOfAromNei] = idx
        except:
            ##toremove
            toRm.append(idx)
    for i in toRm:
        idxOfWantedAtmType.remove(i)
    ri = mol.GetRingInfo()
    riAtoms = ri.AtomRings() 
    return [ _ringIdxToAtmSymbols(_getRingSystem(idx, riAtoms), symbols ) for idx in idxOfNeiOfWantedAtmType], idxOfNeiOfWantedAtmType

def _ringIdxToAtmSymbols(ringSystem, symbols, asStr=True ):
    #ringSystem = [list, [list, ...], ....]
    r1=[symbols[x] for x in  ringSystem[0]]
    if asStr:
        r1=[ x+str(r1.count(x)) for x in sorted(set(r1))]
    res= [r1, ]
    if len(ringSystem)>1:
        for ringLevel in ringSystem[1:]:
            thisLevel = [ [symbols[idx] for idx in ring] for ring in ringLevel] 
            if asStr:
                thisLevel = [ [ x+str(ring.count(x)) for x in sorted(set(ring))] for ring in thisLevel]
            res.append(thisLevel)
    return res





def getRxClass(halos, borons, fullDane, printStat=False):
    lh=len(halos)
    lb=len(borons)
    if lh != lb: raise
    statB=dict()
    statBr=dict()
    statRingBr=dict()
    statRingB=dict()
    allRxNames=[]
    
    onerx=[]
    
    bRingTypeName=[]
    xRingTypeName=[]
    for i in range(lh):
        brRing = []
        brRingName=[]
        for s in halos[i]:
            try:
                usedHalogen='Cl'
                if 'Br' in s:
                    usedHalogen='Br'
                if 'I' in s:
                    if 'I+' in s and s.count('I+') == s.count('I'):
                        pass
                    else:
                        usedHalogen='I'
                ri, neiOfHalogen =getRingType(s, wantedAtmType=usedHalogen)
                brRing.append(ri)
                #print("RI", ri[0][0])
                rname = getRingNames(Chem.MolFromSmiles(s), usedHalogen, neiOfHalogen) 
                #print("HALOGEN", s, usedHalogen, neiOfHalogen, rname)
                brRingName.extend(rname.keys())
                key='other'
                if len(rname)>1:
                    print("RNhalo", rname)
                    key=frozenset( list(rname.keys()) )
                elif len(rname) == 1:
                    key=list(rname.keys())[0]
                if not key in statRingBr:
                    statRingBr[key]=0
                statRingBr[key]+=1
                #print("RI ng type", ri, s , usedHalogen)
                rid = tuple( ri[0][0])
                if not rid in statBr:
                    statBr[rid]=0
                statBr[rid]+=1
                if key=='other': print("OTHERBORON", s)
            except:
                raise
                print("halo problem", s )
        bRing = []
        bRingName=[]
        for s in borons[i]:
            try:
                ri, neiOfBoron = getRingType(s, wantedAtmType='B') 
                bRing.append(ri)
                rid = tuple(ri[0][0])
                #print("RI", ri, "RID", rid)
                #if not rid in statB:
                #    statB[rid]=0
                #statB[rid]+=1
                rname = getRingNames(Chem.MolFromSmiles(s), 'B', neiOfBoron) 
                bRingName.extend(rname.keys())
                #print("BORON", s, rname)
                key='other'
                if len(rname)>1:
                    print("RNboro", rname)
                    key=frozenset( list(rname.keys()) )
                elif len(rname) == 1:
                    key=list(rname.keys())[0]
                if not key in statRingB:
                    statRingB[key]=0
                statRingB[key]+=1
                
            except:
                raise
                print("B problem", s)
        #print ("ONERX", len(brRingName), len(bRingName), '.'.join(brRingName ), '.'.join( bRingName) )
        brNT='other'
        if brRingName:
            brNT='.'.join(brRingName)
        bNT='other'
        if bNT:
            bNT='.'.join(bRingName)
        allRxNames.append( (brNT, bNT) )
        bRingTypeName.append(bNT)
        xRingTypeName.append(brNT)
        ## getFutherinfo:
        #'Pd', 'solvent', 'base', 'ligand', 'special', 'temp', 'time', 'raw', 'yield'
        brGenTyp = list(set([x.split(':::')[0] for x in brRingName]))
        bGenTyp = list(set([x.split(':::')[0] for x in bRingName]))
        #print ("FF", fullDane['raw'][0]['rxInfo']['sbs'] )
        #raise
        onerx.append( json.dumps( ("ONERX", brGenTyp, bGenTyp, brRingName, bRingName,  fullDane['solvent'][i],  fullDane['base'][i], fullDane['temp'][i], fullDane['ligand'][i], 
          fullDane['Pd'][i], fullDane['special'][i].split('; '), fullDane['yield'][i], fullDane['litSource'][i], fullDane['raw'][i]['rxInfo']['sbs'], fullDane['raw'][i]["Reaction"],
           fullDane['raw'][i]['rxInfo'] ) ) )
    #print(halos, borons)
    if printStat:
        print("STAT B", statB)
        print("STAT Br", statBr)
        print("STAT B")
        for i in sorted(statB, key = lambda x:statB[x]):
            print(i, statB[i])
        print("STAT halogen")
        for i in sorted(statBr, key = lambda x:statBr[x]):
            print(i, statBr[i])
        print("STAT RING X")
        for i in sorted(statRingBr, key = lambda x:statRingBr[x]):
            print(i, statRingBr[i])

        print("STAT RING B")
        for i in sorted(statRingB, key = lambda x:statRingB[x]):
            print(i, statRingB[i])
        rxNameStat( allRxNames) 
        #print("TYPES OF BORON")
        sbsNameStat(xRingTypeName, mode='X')
        sbsNameStat(bRingTypeName, mode='B')
    return onerx



def combinateFiles(res, removeDuplicatesByPos=(0,), ):
    header=[ r['header'] for r in res]
    datas = [ r['data'] for r in res]
    for h in header[1:]:
        if h != header[0]:
            print("H", h)
            print("H", header[0])
            raise
    header=header[0]
    data = []
    mustBeUniq={pos:set() for pos in removeDuplicatesByPos}
    for dfile in datas:
        for line in dfile:
            ignoreLine=False
            for pos in mustBeUniq:
                if line[pos] in mustBeUniq[pos]:
                    print("ignore duplicated", header[pos], "which has value:", line[pos])
                    ignoreLine = True
                else:
                    mustBeUniq[pos].add( line[pos] )
            if ignoreLine:
                continue
            if len(line) == len(header):
                data.append(line)
            elif len(line) == len(header) +1:
                if line[-1].strip():
                    raise
                else:
                    data.append(line[:-1])
            else:
                print("LEN", len(line) )
                raise
    #return (header, data)
    return {h:[d[i] for d in data] for i,h in enumerate(header)}



def filterOutNotMatched( data, reactions=('suzuki', ), entryName='Reaction' ):
    toRmIdx=[]
    data['rxInfo']=[]
    for i, rx in enumerate(data[entryName]):
        if 'suzuki' in reactions:
            rxInfo = isSuzuki(rx)
            if not rxInfo:
                toRmIdx.append(i)
                data['rxInfo'].append( rxInfo)
            else:
                data['rxInfo'].append(rxInfo)

    print("TO RM", len(toRmIdx))
    toRmIdx.reverse()
    for i in toRmIdx:
        for header in data:
            removed=data[header].pop(i)
            #if header == 'rxInfo': print("rm info", removed, end=' ')
        #print(i, rx)
    return data


def isPd(text):
    lowe=text.lower()
    if 'Pd' in text or 'pallad' in lowe or 'palad' in lowe:
        return True
    return False

def isBoronic(smi):
    return smi.count('B') > smi.count('Br')

def isHalogen(smi):
    return smi.count('Br') > 0 or smi.count('I') >0 or smi.count('Cl')>0

def countRings(smi):
    if '%' in smi: #more two digit ring numbering
        raise
    bra = re.findall('\[\d+', smi)
    ket = re.findall('\d+\]', smi)
    allDigits= re.findall('\d+', smi) 
    for b in bra:
        allDigits.remove( b[1:])
    for k in ket:
        allDigits.remove( k[:-1])
    #print("allDO", allDigits)
    return sum([len(x) for x in allDigits])/2


def simpleStat(data):
    for header in data:
        notEmpty = [d for d in data[header] if d.strip() ]
        withPd = set([x for x in notEmpty if isPd(x)])
        onlyPd=set()
        for manyEntry in withPd:
            _ = [ onlyPd.add(x.strip() ) for x in manyEntry.split(';') if isPd(x) ]
        if len(withPd) < 100_000:
            print( header, "not empty", len(notEmpty), "uniq", len( set(notEmpty) ), "with Pd", len(withPd), "only", len(onlyPd), onlyPd )
        else:
            print( header, "not empty", len(notEmpty), "uniq", len( set(notEmpty) ), "with Pd", len(withPd) )



def isSuzuki(smiles, verbose=False):
    suzukiRx=AllChem.ReactionFromSmarts('[#6:1][Br,I,Cl:2].[#6:3][BX3:4]([OX2:5])[OX2:6]>>[#6:1][#6:3].[*:2].[B:4]([O:5])[O:6]')
    history=dict()
    try:
        substrates, products= smiles.split('>>')
    except:
        if not smiles.strip():
            return False
        print("PROBLEM WITH", smiles)
    substrates = substrates.split('.')
    products = products.split('.')
    boronicInitial = set([ s for s in substrates if isBoronic(s)])
    halogenInitial = set([ s for s in substrates if isHalogen(s)])
    prodMols = []
    for p in products:
        #print("P", p, countRings(p), smiles )
        if countRings(p) < 2:
            continue
        try:
            mol=Chem.MolFromSmiles(p)
            if not mol:
                print("no mol from p")
                raise
            prodMols.append(mol)
        except:
            print("cannot proceed smiles", p)
            #raise
    canonProd = [Chem.MolToSmiles(s) for s in prodMols]
    if not canonProd:
        #print("no prod in ", smiles)
        return False
    canonProdNoStereo = [Chem.MolToSmiles(s, False) for s in prodMols]
    maxIter = 10
    halogen = halogenInitial
    boronic = boronicInitial
    for i in range(maxIter):
        res, resNoStereo = makeAllCombination( suzukiRx, [tuple(halogen), tuple(boronic)])
        if any([p in canonProd for p in res] )  or any([p in canonProdNoStereo for p in resNoStereo ]):
            obtainedTrueProductNoStereo = [p for p in resNoStereo if p in canonProdNoStereo]
            obtainedTrueProduct = [p for p in res if p in canonProd]
            substrateForTrueProduct = {p:res[p] for p in res}
            substrateForTrueProductNoStereo = {p:resNoStereo[p] for p in resNoStereo}
            #print("REN", resNoStereo)
            allHalo =set()
            for pr in resNoStereo:
                _ = [ allHalo.add(s[0]) for s in resNoStereo[pr ] ]
            allBoro =set()
            for pr in resNoStereo:
                _ = [ allBoro.add(s[1]) for s in resNoStereo[pr] ]
            return {'products':tuple(obtainedTrueProduct), 'productsNoStereo':tuple(obtainedTrueProductNoStereo), 
                'halogens':tuple([h for h in halogenInitial if h in allHalo]), 
                'borons':tuple([b for b in boronicInitial if b in allBoro]), 
                'sbs':substrateForTrueProduct, 'sbsNoStereo':substrateForTrueProductNoStereo, 'history':history }
            #return True
        halo = set([s for s in res if isHalogen(s)])
        boro = set([s for s in res if isBoronic(s)])
        for h in halo:
            if not h in history:
                history[h]=[]
            history[h].extend(res[h])
        for b in boro:
            if not b in history:
                history[b] =[]
            history[b].extend(res[b])
        if halo and boro:
            if verbose:
                print("HALOGEN", halogen, boronic)
                print("HALO BORO", halo, boro, "\n canonProd:",canonProd, "\nsmiles", smiles)
                print("RES", res, resNoStereo)
                print("EXP", canonProd, canonProdNoStereo)
            return False
            #raise
        elif halo:
            halogen = halo
        elif boro:
            boronic = boro
        else:
            return False
    return None


def makeAllCombination( reactionObj, substratesListList):
    allProds = defaultdict(set)
    allProdsNoStereo = defaultdict(set)
    for sbsList in itertools.product(*substratesListList):
        #print("SS", sbsList, "SS", substratesListList)
        sbs = [Chem.MolFromSmiles(s) for s in sbsList]
        if any([s ==None for s in sbs]):
            print("sbsList", sbsList)
            continue
        rxProds = [ x[0] for x in reactionObj.RunReactants(sbs) if x]
        _ = [ Chem.SanitizeMol(x) for x in rxProds]
        _ = [ allProds[Chem.MolToSmiles(m, True)].add( tuple(sbsList) ) for m in rxProds]
        _ = [allProdsNoStereo[ Chem.MolToSmiles(m, False) ].add( tuple(sbsList) ) for m in rxProds ]
    allProds = {p:tuple(allProds[p]) for p in allProds}
    allProdsNoStereo = {p:tuple(allProdsNoStereo[p]) for p in allProdsNoStereo}
    return allProds, allProdsNoStereo


def findPd(data, pos, ignoredHeader={'Fulltext of reaction', 'Example title'}):
    withPd=[ ]
    #entries =[ data[head][lid] for head in data.keys() if isPd(data[head][lid]) ]
    for header in data.keys():
        if header in {'Fulltext of reaction', 'Example title', 'rxInfo'}:
            continue
        entry= data[header][pos]
        _ = [ withPd.append( e ) for e in entry.split('; ') if isPd(e) ]
    ##make canonical name:
    canonName={'tetrakis(triphenylphosphine) palladium(0)':'Pd[P(Ph)3]4', '1: tetrakis(triphenylphosphine) palladium(0)':'Pd[P(Ph)3]4', 'tetrakis(triphenylphosphine) palladium(0)':'Pd[P(Ph)3]4',
        'tetrakis (triphenylphosphine) palladium (0)':'Pd[P(Ph)3]4','tetrakistriphenylphosphanepalladium(0)':'Pd[P(Ph)3]4', 'tetrakis(triphenylphosphine)palladium (0)':'Pd[P(Ph)3]4', 
        'tetrakis(triphenylphosphine) palladium(0) / N,N-dimethyl-formamide':'Pd[P(Ph)3]4', 'tetrakis-(triphenylphosphino)-palladium(0)':'Pd[P(Ph)3]4',
        'tetrakistriphenylphosphanepalladium(0)':'Pd[P(Ph)3]4', 'tetrakis (triphenylphosphine) palladium (0)':'Pd[P(Ph)3]4', 'tetrakis(triphenylphosphine)palladium (0)':'Pd[P(Ph)3]4',
         'tetrakis(triphenylphosphine) palladium(0) / N,N-dimethyl-formamide':'Pd[P(Ph)3]4', 'tetrakis-(triphenylphosphino)-palladium(0)':'Pd[P(Ph)3]4',
        'bis(tri-tert-butylphosphine)palladium(0)':'Pd[P(Ph)3]4',

        '[1,1-bis(diphenylphosphino)ferrocene]-dichloropalladium': 'Pd(dppf)Cl2', "[1,1'-bis(diphenylphosphino)ferrocene]dichloropalladium(II)": 'Pd(dppf)Cl2', 
        "(1,1'-bis(diphenylphosphino)ferrocene)palladium(II) dichloride":'Pd(dppf)Cl2', "(1,1'-bis(diphenylphosphino)ferrocene)palladium(II) dichloride / 1,4-dioxane":'Pd(dppf)Cl2',
        "1,1'-bis(diphenylphosphino)ferrocene-palladium(II)dichloride dichloromethane complex":'Pd(dppf)Cl2',
        "1: sodium carbonate / (1,1'-bis(diphenylphosphino)ferrocene)palladium(II) dichloride / DMF (N,N-dimethyl-formamide)":'Pd(dppf)Cl2',
        '1: sodium carbonate / tetrakis(triphenylphosphine) palladium(0) / 1,2-dimethoxyethane':'Pd(dppf)Cl2', "dichloro(1,1'-bis(diphenylphosphanyl)ferrocene)palladium(II)*CH2Cl2":'Pd(dppf)Cl2',
        'palladium bis[bis(diphenylphosphino)ferrocene] dichloride':'Pd(dppf)Cl2',

        'bis-triphenylphosphine-palladium(II) chloride':'Pd[P(Ph)3]2Cl2', 'bis(triphenylphosphine)palladium(II) chloride':'Pd[P(Ph)3]2Cl2', 'bis(triphenylphosphine)palladium(II)-chloride':'Pd[P(Ph)3]2Cl2',
        'bis-triphenylphosphine-palladium(II) chloride':'Pd[P(Ph)3]2Cl2', 'bis(triphenylphosphine)palladium(II) dichloride':'Pd[P(Ph)3]2Cl2', 'dichlorobis(triphenylphosphine)palladium[II]':'Pd[P(Ph)3]2Cl2',
        'trans-bis(triphenylphosphine)palladium dichloride':'Pd[P(Ph)3]2Cl2', 'dichlorobis(triphenylphosphine)palladium(II)':'Pd[P(Ph)3]2Cl2',

        'tris-(dibenzylideneacetone)dipalladium(0)':'Pd2(dba)3', 'bis(dibenzylideneacetone)-palladium(0)':'Pd2(dba)3', 'tris(dibenzylideneacetone)dipalladium(0) chloroform complex':'Pd2(dba)3', 
        'tris(dibenzylideneacetone)dipalladium (0)':'Pd2(dba)3',  '"tris-(dibenzylideneacetone)dipalladium(0)':'Pd2(dba)3',

        'bis(di-tert-?butyl(4-?dimethylaminophenyl)?phosphine)?dichloropalladium(II)': 'Pd(amphos)Cl2', 'bis[di-t-butyl(p-dimethylaminophenyl)phosphino]palladium (II) Dichloride':'Pd(amphos)Cl2',
        
        '[1,3-bis(2,6-diisopropylphenyl)imidazol-2-ylidene](3-chloropyridyl)palladium(ll) dichloride':'PEPPSI-IPr-PdCl2', '[1,3-bis(2,6-diisopropylphenyl)imidazol-2-ylidene](3-chloropyridyl)palladium(II) dichloride':'PEPPSI-IPr-PdCl2', 
        '[1,3-bis(2,6-diisopropylphenyl)imidazol-2-ylidene](3chloro-pyridyl)palladium(II) dichloride':'PEPPSI-IPr-PdCl2', 
    }
    
    toRet= []
    for pd in withPd:
        if pd in canonName:
            toRet.append( canonName[pd] )
        else:
            toRet.append(pd)
    #withPd =[ canonName[x] for x in withPd]
    print("WITHPD", toRet)
    return list(set(toRet))


def findSolvent(data, pos):
    solvents= [x.strip() for x in data['Solvent (Reaction Details)'][pos].split(';') if x]
    if solvents:
        return sorted( set(solvents) )
    return solvents

def findTemp(data, pos):
    tmp = data['Temperature (Reaction Details) [C]'][pos]
    temps=[]
    for t in tmp.split():
        try:
            temps.append( int(t) )
        except:
            continue
    return temps

def findTime(data, pos):
    rxtime = data['Time (Reaction Details) [h]'][pos]
    #print(rxtime)
    return rxtime

def findSpecialCare(data,pos):
    oth=data['Other Conditions'][pos]
    #print(oth)
    return oth

def findBase(data, pos):
    allowedHeader={'Reactant','Reagent'}
    #excludedHeader={'Links to Reaxys', 'Reaction: Links to Reaxys'}
    expectedKeywords=('caesium', 'carbonate', 'fluoride', 'hydroxide', 'amine', 'ammonium', 'phosphate', 'anolate')
    #allowedKeywords=('acetate')
    addit=[]
    for header in data:
        if not header in allowedHeader:
            continue
        for entry in data[header][pos].split(';'):
            entry=re.sub(' [a-z]*hydrate', '', entry)
            entry=entry.replace('cesium', 'caesium').replace('tribase', '').replace('barium(II)', 'barium').replace('"', '').replace(' n ', '')
            entry=entry.replace('barium dihydroxide', 'barium hydroxide').replace('tribasic', '')
            entry=entry.strip()
            if 'pallad' in entry or 'bromo' in entry or 'iodo' in entry or 'ammonium' in entry or ' acid' in entry or 'iodine' in entry: continue
            if 'Bromo' in entry or 'midazolium hexafluorophosphate' in entry: continue
            if any( [k in entry for k in expectedKeywords]):
                addit.append( entry )
            if 'acetate' in entry and not 'pallad' in entry:
                addit.append(entry.strip() )
    if addit:
        return addit
        
    else:
        print("nobase", [data[x][pos] for x in data] )
    return ()



def findLigand(data, pos, pd):
    headerToIgnore={'Fulltext of reaction', 'References', 'Product', 'rxInfo'}
    found=[]
    toIgnore= {'1-(Di-tert-butyl-phosphinoyl)-2-iodo-benzene', 'phosphoric acid', 'diethyl 1-(2-bromophenyl)-3-oxopropylphosphonate', 'General procedure for Suzuki coupling in position 2 (main text Scheme 6, method A).'}
    for header in data:
        if header in headerToIgnore:
            continue
        entries= data[header][pos]
        for entry in entries.split(';'):
            if 'pos' in entry or 'phos' in entry:
                if 'Pd' in entry or 'palladium' in entry or 'phosphate' in entry:
                    continue
                entry= entry.strip()
                if entry in ('triphenylphosphine', 'triphenylphosphine / 1,4-dioxane'):
                    entry='PPh3'
                if entry in {"1,1'-bis(diphenylphosphino)ferrocene", "1,1'-bis-(diphenylphosphino)ferrocene"}:
                    entry='dppf'
                if entry in ('1,4-di(diphenylphosphino)-butane', ):
                    entry = 'dppb'
                if entry in {'tris-(m-sulfonatophenyl)phosphine', 'triphenylphosphine-3,3?,3?-trisulfonic acid trisodium salt', 'triphenylphosphine trisulfonic acid', 
                            'trisodium tris(3-sulfophenyl)phosphine', }:
                    entry='TPPTS'
                if entry in {"tricyclohexylphosphine", "tricyclohexylphosphine tetrafluoroborate", "tricyclohexylphosphine tetrafluoroborate", 
                            "tris(cyclohexyl)phosphonium tetrafluoroborate"}:
                    entry = 'P(cychex)3'
                if entry in ('1,2-bis-(diphenylphosphino)ethane'):
                    entry='dppe'
                if entry in ('tributylphosphine'):
                    entry='PBu3'
                if entry in ('tri-tert-butyl phosphine', "tris[tert-butyl]phosphonium tetrafluoroborate", "tri-t-butylphosphonium tetraphenylborate complex", 
                        "tri tert-butylphosphoniumtetrafluoroborate", "tri tert-butylphosphoniumtetrafluoroborate", "tri-tertiary-butyl phosphonium tetrafluoroborate",
                        "tri-tert-butylphosphonium tetrafluoroborate"):
                    entry='PtBu3'
                if entry in ('tris-(o-tolyl)phosphine', 'tris(2-methylphenyl)phosphine'):
                    entry='P(o-Tol)3'
                if entry in ("dicyclohexyl-(2',6'-dimethoxybiphenyl-2-yl)-phosphane", "dicyclohexyl(2?,6?-dimethoxy-[1,1?-biphenyl]-3-yl)phosphine", "2-dicyclohexylphosphino-2?,6?-dimethoxybiphenyl",
                        "2-dicyclohexylphosphino-2?,6?-diisopropoxy-1,1?-biphenyl"):
                    entry='SPhos'
                if entry in ('(4-(N,N-dimethylamino)phenyl)-di-tert-butylphosphine'):
                    entry ='APhos'
                if entry in ('4,5-bis(diphenylphos4,5-bis(diphenylphosphino)-9,9-dimethylxanthenephino)-9,9-dimethylxanthene'):
                    entry='xantphos'
                if entry in ("(1RS,2RS,3SR,4SR)-1,2,3,4-tetrakis((diphenylphosphanyl)methyl)cyclopentane",):
                    entry = 'tedicyp'
                if entry.endswith('oxide'):
                    continue
                if not entry in toIgnore:
                    found.append( entry)
    if pd:
        if any(["P(Ph)3" in pdentry or 'PPh3' in pdentry for pdentry in pd]):
            found.append("PPh3")
        if any(['dppf' in pdentry for pdentry in pd]):
            found.append('dppf')
    found = list(set(found))
    print("LIGANDS", *found, sep='\t')
    return found


def findYield(data,pos):
    #print( len(data), type(data), data.keys() )
    #a= input('xx')
    try:
        return tuple(map(float, data['Yield (numerical)'][pos].split(';')))
    except:
        if  data['Yield'][pos] or  data['Yield (numerical)'][pos] or  data['Yield (optical)'][pos]:
            print("YIELD1", data['Yield'][pos])
            print("YIELD2", data['Yield (numerical)'][pos])
            print("YIELD3", data['Yield (optical)'][pos])
        return ""

def findSource(data, pos):
    ref= data['References'][pos] 
    if 'Patent' in ref:
        return ('PATENT', ref)
    if 'Article' in ref:
        return ('ARTICLE', ref)
    return ("OTHER", ref)

def getCNR(data,pos):
    link = data['Reaction: Links to Reaxys'][pos]
    rxid = link.split('RX.ID=')[1].split('&')[0]
    link = data['Links to Reaxys'][pos]
    cnr = link.split('CNR=')[1].split('&')[0]
    return str(rxid), str(cnr)

def getCorrection( cdict):
    corr=dict()
    for key in cdict:
        corr[key]=dict()
        for line in open(cdict[key]):
            rxid, cdn, value = line.split('\t')[:3]
            corr[key][ (str(rxid), str(cdn) ) ] =value
    return corr


def entryStat(data, limits, removeDuplicate=True, correctionFiles=None ):
    correction=False
    onlyWithYield = limits.withyieldonly
    if correctionFiles:
        correction=getCorrection(correctionFiles)
    numEntry = len(data[ tuple(data.keys())[0] ])
    ##temp, time, base, solvent, Pd (ligand)
    uniqSet= set()
    dane = {'Pd':[], 'solvent':[], 'base':[], 'ligand':[], 'special':[], 'temp':[], 'time':[], 'raw':[], 'yield':[], 'litSource':[] }
    #print("NOBASE", '\t'.join([k for k in data]), sep='\t' )
    for lid in range(numEntry):
        withPd=findPd(data, lid)
        #print("pd", withPd)
        solvent=findSolvent(data, lid)
        temp = findTemp(data,lid)
        time = findTime(data,lid)
        special = findSpecialCare(data,lid)
        base= findBase(data,lid)

        rxyield = findYield(data, lid)
        rxidcnr = getCNR(data,lid)
        if correction:
            if 'base' in correction and rxidcnr in correction['base']:
                base = correction['base'][rxidcnr].strip()
                if 'o base' in base or 'cant find paper' in base or 'not suzuki' in base: continue
                if ' or ' in base:
                    base = base.split(' or ')[0]
                base = base.split(', ')
            print("NO BASE", base) #, "\n", [(x,data[x][lid]) for x in data])
            #raise
            if 'ligand' in correction and rxidcnr in correction['ligand']:
                ligand = correction['ligand'][rxidcnr].strip()
                if 'exclude' in ligand or 'bad data' in ligand:
                    continue
        ligand= findLigand(data, lid, withPd)
        if not base and limits.withbaseonly:
            continue
        if onlyWithYield and not rxyield:
            continue
        if not ligand and limits.withligandonly:
            continue
        if not temp and limits.withtemponly:
            continue
        if not solvent and limits.withsolvonly:
            continue
        if not withPd and limits.withpdonly:
            continue
        if removeDuplicate:
            thisId = (tuple(withPd), tuple(ligand), tuple(base), tuple(solvent), str(data['rxInfo'][lid]['sbs']) )
            if thisId in uniqSet:
                continue
            uniqSet.add( thisId)
        print("LIGAND==", ligand, rxidcnr, rxidcnr in correction['ligand'], "PD", withPd)
        dane['Pd'].append( tuple(withPd))
        dane['solvent'].append(solvent)
        dane['base'].append(base)
        dane['ligand'].append(ligand)
        dane['special'].append(special)
        dane['temp'].append(temp)
        dane['time'].append(time)
        dane['yield'].append(rxyield)
        dane['raw'].append({k:data[k][lid] for k in data})
        dane['litSource'].append( findSource(data,lid) )
        #if not ligand and not( 'Pd[P(Ph)3]4' in withPd):
        #    if any(['phos' in pdcat or 'Pd(dppf)Cl2' in pdcat or 'Pd[P(Ph)3]2Cl2' in pdcat  for pdcat in withPd]): continue
        #    print( 'NOLIGAND:', withPd,  '\t'.join([str(data[k][lid]) for k in data]), sep='\t'  )
        #if not base:
        #    print("NOBASE", '\t'.join([str(data[k][lid]) for k in data]), sep='\t'  )
    return dane
        #print("Pd", withPd, 'S:', solvent, 'base', base, 'L:', ligand, )
        #    print(withPd)



if __name__ == "__main__":
    parser=parseArgs()
    print("P", parser)
    files=[]
    if parser.heterohetero:
        prefix='downloadedRx/hetero-hetero/'
        heterohetero=['39074585_20200310_190414_098.xls', '39074585_20200310_191650_204.xls', '39074585_20200310_194241_493.xls', 'Suzuki_Har_1-2500.csv', 'Suzuki_Har_2501-5000.csv']
        for i in heterohetero:
            files.append( prefix+i)
    if parser.arylhetero:
        prefix='downloadedRx/hetero-aryl/'
        arylhetero=['aga1.csv', 'aga2.csv', 'aga3.csv', 'aga4.csv', 'aga5.csv',]
        for i in arylhetero:
            files.append(prefix+i)
    if parser.arylaryl:
        prefix='downloadedRx/aryl-aryl/'
        arylaryl=['Reaxys_Exp_20200424_184807.csv', 'Reaxys_Exp_20200424_201155.csv', 'Reaxys_Exp_20200425_011430.csv', 'Reaxys_Exp_20200425_060051.csv', 'Reaxys_Exp_20200427_151519.csv']
        for i in arylaryl:
            files.append(prefix+i)
    res = [ parseFile(fn, includePatents=parser.withpatents) for fn in files]
    print( [ len(r['header']) for r in res])
    #def combinateFiles(res, removeDuplicatesByPos=(0,), ):
    data= combinateFiles(res, removeDuplicatesByPos=[] )
    print("DATA", data.keys())
    data=filterOutNotMatched( data, reactions=('suzuki', ), entryName='Reaction' )
    print("DATA", data.keys())
    print("+++++++++++++++++")
    #simpleStat(data)
    dane=entryStat(data, parser, removeDuplicate=True,  correctionFiles={'base':'./downloadedRx/nobase.csv', 'ligand':'./downloadedRx/noligand.csv'} )
    import json
    json.dump(dane, open(parser.output, 'w') )
    print("h", data.keys() )
    allrx=getRxClass( [lst['rxInfo']['halogens'] for lst in dane['raw']], [ lst['rxInfo']['borons'] for lst in dane['raw'] ], dane)
    fnw= open( parser.output+'.onerx', 'w')
    for i in allrx:
        print(i, file=fnw)
    fnw.close()