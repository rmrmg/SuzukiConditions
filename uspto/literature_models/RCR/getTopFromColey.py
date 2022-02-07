# -*- coding: utf-8 -*-
import sys
import makeReactionFromParsed


def parseLogFile(fn, smiSolver, ref, args):
    inpfile = open(fn).readlines()
    head = {x:i for i,x in enumerate(inpfile[0].strip().split(';'))}
    uniqBases = {}
    uniqSolv = {}
    tops = dict()
    for line in inpfile[1:]:
        if not ';' in line:
            continue
        line = line.strip().split(';')
        base = line[ head['base']]
        solvent = line[ head['solvent']]
        rxn = line[ head['smiles'] ]
        if not rxn in ref:
            raise
        if not base:
            #print("NOT BASE", line)
            #continue
            bklas == 'other'
        else:
            bklas = smiSolver.detectBaseClass( set(base.split('.')) )
        klas= smiSolver.getNewSolventClass(set(solvent.split('.')))
        if args['ignoreOtherSolvent'] and ref[rxn][1] == 'other':
            continue
        if not rxn in tops:
            tops[rxn] = {999, }

        if  ref[rxn] == [bklas, klas]:
            print("top::", rxn, line[ head['topK']], ref[rxn])

            tops[rxn].add( int(line[ head['topK']]) )
        for b in base.split('.'):
            if not b in uniqBases:
                uniqBases[b] = 0
            uniqBases[b] +=1
        for s in solvent.split('.'):
            if not s in uniqSolv:
                uniqSolv[s] = 0
            uniqSolv[s] += 1
    #print("S", uniqSolv)
    #print("B", uniqBases)
    #return uniqSolv, uniqBases
    return tops

if __name__ == "__main__":
    files = ['coley_preds_top10.csv', ]# 'forrecomp2_valid0.txt']
    smiSolver = makeReactionFromParsed.SmilesResolver()
    args = {
        'ignoreOtherSolvent':False,
    }
    #inpfile = 'inputy_coley/rxcondr_oldclass_valid0.txt'
    inpfile = sys.argv[1]
    #inpfile = 'inputy_coley/rxcondr_newclass_valid0.txt' #new classes aka coarse-grained class 
    ref = {x.split(';')[0]:x.strip().split(';')[2:] for x in open(inpfile).readlines()[1:] }
    for fn in files:
        print("FN", fn)
        tops = parseLogFile(fn, smiSolver, ref, args)
        for i in range(1,11):
            thisTop = len([t for t in tops if i == min(tops[t])] )
            print("top",i, (thisTop*100)/len(tops), thisTop, "of", len(tops))