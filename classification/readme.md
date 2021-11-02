# Scripts and logs from classification task
## Information about files and directories:
- autoencoder/ - directory with scripts and logs from autoencoder training and hyperparameters optimization
- logs/ - directory with logs from NN training 
- classif_v5desc.py - script for classification using smiles of coupling partners and encoded information about ligand and temperature. 
 Model parameters can be set using following options:
  - --n1 number of neurons in first hidden layer
  - --n2 number of neurons in second hidden layer
  - --act1 activation function after first hidden layer
  - --act2 activation function after second hidden layer
  - --rep how information about coupling partner should be encoded. There are 3 allowed values: 
    - morgan - morgan fingerprint
    - desc - RDkit's descriptors
    - morgandesc - concatenation of morgan fingerprint and rdkit's descriptors

  The script expects following input format:
 `baseClass      solventClass        smiles1         smiles2         ligandAndTemperature              Yield`
Where baseClass and solventClass are number (integer) of base/solvent class, smiles1 and smiles2 are smiles of coupling partners, 
ligandAndTemperature is description of ligand (and temperature) can have many column. Last column is numeric value of reaction yield. 
The field need to be separated by tab.

- classif_v5smi.py - script for training clasification similar to `classif_v5desc.py` the only difference is this scripts use coupling partners representations 
 fromautoencoder. The representation should be provided with options `-encdict ENCDICT` where ENCDICT is file json  with dictionary
 of form 'smiles':'smiles_representation_from_encoder'. Format of input file is the same as for `classif_v5desc.py`.

- keras_embedlig_v3new.py - train classification using embeding representation of solvent, base, and ligand. Expected input format is
 `solvent base ligand1 ligand2 temperature substrate1 substrate2 yield` where solvent and base is class of used solvent/base provided as integer, ligand1, 
  ligand2 are numeric representation of ligand(s) if only one ligand was used ligand2 is 0, substrate1 and substrate2 is encoded representation of coupling partner.
  By default the script exect 6 solvent classes, 7 base classes, 81 ligand classes (in our case just enumeration of detected ligands) and 512 columns length 
  representation of each coupling partner. The values can be adjusted in makeModel() function.
