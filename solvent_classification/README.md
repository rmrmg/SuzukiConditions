# Requirements

- sklearn
- Mol2Vec
- tensorflow 2.0
- keras

# Workflow to reproduce Table 3.

1. Compute descriptor file `processed_data_sml_morgan_rdkit_m2v.pkz`
`python compute_descriptors.py --datafile your_DATAFILE.onerx --mol2vec_model path_to_mol2vec_model`

2. 
