import pandas as pd
import pickle, sys
from Reaction_condition_recommendation.scripts.neuralnetwork import NeuralNetContextRecommender
model_json = 'Reaction_condition_recommendation/models_downloaded/model.json'
info_path = 'Reaction_condition_recommendation/models_downloaded/'
model_weights = 'Reaction_condition_recommendation/models_downloaded/weights.h5'

cont = NeuralNetContextRecommender()
cont.load_nn_model(model_path=model_json, info_path=info_path, weights_path=model_weights)

dataset = pd.read_csv(sys.argv[1], sep=';')
#    dataset['reaction_smiles'] = dataset['bromide'] + '.' + dataset['boronate'] + '>>' + product


data = dict(rr_row_idx=[], smiles=[], topK=[], temperature=[], solvent=[], base=[], catalyst=[], score=[])
for rrid, row in dataset.iterrows():

    try:
        preds, scores = cont.get_n_conditions(row['reaction_smiles'], 10, with_smiles=False, return_scores=True)
    except:
        scores = [-1]
        preds = [[None, '', '', '']]
        print('fail for ',row['reaction_smiles'])

    for j, x in enumerate(preds):
        t, solvent, base, catalyst = x[:4]
        score = scores[j]
        data['rr_row_idx'].append(rrid)
        data['smiles'].append(row['reaction_smiles'])
        data['topK'].append(j+1)
        data['temperature'].append(t)
        data['solvent'].append(solvent)
        data['base'].append(base)
        data['catalyst'].append(catalyst)
        data['score'].append(score)
    if (rrid+1)%100==0:
        print(rrid+1, 'done')
    if (rrid+1)%1000==0:
        with open('backup_tmp.pkl','wb') as f:
            pickle.dump(data, f)
        print('checkpoint saved')
data = pd.DataFrame(data)
data.to_csv('coley_preds_top10.csv',sep=';', index=False)

# print(cont.name_to_category('c1','Reaxys Name palladium on activated charcoal'))
