import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--datafile', type=str, default='heteroaryl_suzuki.onerx')
parser.add_argument('--output', default='heteroaryl_suzuki.csv')
args= parser.parse_args()
import pandas as pd

fname=args.datafile
sidx = 5
bidx = 6
tidx = 7
lidx = 9
yidx = 11
aidx = 12
sml_idx = 14

data = {'solvent':[], 'base':[], 'temperature':[], 'ligand':[], 'yield':[], 'article':[], 'reaction_smiles':[]}

i=0
with open(fname, 'r') as f:
    for line in f:
        x = eval(line)
        data['solvent'].append(',,'.join(x[sidx]))
        data['base'].append(',,'.join(x[bidx]))
        data['temperature'].append(x[tidx][0] if x[tidx]!=[] else None)
        data['ligand'].append(',,'.join(x[lidx]))
        data['yield'].append(x[yidx][0] if x[yidx]!=[] else None)
        data['reaction_smiles'].append(x[sml_idx])
        data['article'].append(x[aidx][1] if len(x[aidx])>1 else '')
        i+=1
data = pd.DataFrame(data)
data.to_csv(args.output, sep=';', index=False)
