import argparse
import pandas
import pkg_resources
import torch
from rxnfp.models import SmilesClassificationModel
#based on https://github.com/rxn4chemistry/rxn_yields/blob/master/nbs/05_model_training.ipynb
# data
def parseArgs():
    parser = argparse.ArgumentParser(description='train yieldBERT')
    parser.add_argument('--train', type=str, required=True, help='training set')
    parser.add_argument('--test', type=str, required=True, help='test set')
    parser.add_argument('--out', type=str, required=True, help='directory wihere results will be stored')
    args = parser.parse_args()
    return args

args = parseArgs()

model_args = {'num_train_epochs': 30, 'overwrite_output_dir': True,
    'learning_rate': 0.00009659, #oryg
    # 'learning_rate': 0.000059659,
    'gradient_accumulation_steps': 1, 'regression': True, "num_labels":1, "fp16": False,
    "evaluate_during_training": True, 'manual_seed': 42,
    "max_seq_length": 300, "train_batch_size": 16,"warmup_ratio": 0.00,
    "config" : { 'hidden_dropout_prob': 0.7987 } #oryg
    #"config" : { 'hidden_dropout_prob': 0.5 }
}


model_path =  pkg_resources.resource_filename(
                "rxnfp",
                f"models/transformers/bert-best-1024/" # change pretrained to ft to start from the other base model
)
print("MODE", model_path)
yield_bert = SmilesClassificationModel("bert", model_path, num_labels=1, 
                                       args=model_args, use_cuda=torch.cuda.is_available())
dftrain = pandas.read_csv(args.train, delimiter=' ', header=None)
dftest = pandas.read_csv(args.test, delimiter=' ', header=None)
#train_df = dftrain.iloc[:][['rxn', 'y']] 
#test_df = dftest.iloc[:][['rxn', 'y']] #
train_df = dftrain
test_df = dftest
train_df.columns = ['text', 'labels']
test_df.columns = ['text', 'labels']
print("TRAIN", train_df.shape, "LABEL", train_df.labels)
print("TR", train_df)
#raise
mean = train_df.labels.mean()
std = train_df.labels.std()
print("MEAN", mean, "STD", std)
train_df['labels'] = (train_df['labels'] - mean) / std
test_df['labels'] = (test_df['labels'] - mean) / std
train_df.head()

yield_bert.train_model(train_df, output_dir=args.out, eval_df=test_df)
#print(type(res), res)