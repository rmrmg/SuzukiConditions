# Requirements

- sklearn
- Mol2Vec
- tensorflow 2.0
- keras

# Descriptor file.

Compute descriptor file `processed_data_sml_morgan_rdkit_m2v.pkz`

`python compute_descriptors.py --datafile your_DATAFILE.onerx --mol2vec_model path_to_mol2vec_model`

# Reproduction of Table 2.

- ECFP6 of substrates with joined base

`python conditions_mlp_cv2.py --desc ecfp6 --log fgp_with_base_default_std.log --standardize --join_base`

- Mol2Vec

`python conditions_mlp_cv2.py --desc m2v --log m2v_default_std.log --standardize`


