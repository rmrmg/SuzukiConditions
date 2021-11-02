# Scripts and training logs for regression  task (yield prediction)

## scripts:
- regressor.py - script for regression problem. The script requires data in format `baseClass solventClass substratesRepresentation yield`. Field need to be separated by tab and substrateRepresentation can be any length and any time (this will be directly passed to the model).

- keras_embedingyield_v3new.py  - as above but use embeding layers and input in format 'solvent1  solvent2 solvent3 solvent4 base1 base2 ligand1 ligand2 temperature substratesRepresentation yield", where is  solvent*, base*, and ligand* fields are single number representation of solvent, base or ligand. Such representations allow to store up to 4 different solvents, and up to 2 different bases nad ligands. Substrates representation is 512+512 morgan fingerprint of both coupling partners.

- regresMclass_new.py - regression with with additional penalty (custom1 and custom2 in the paper)
- regresMclass_new_2penalty_Ynorm.py  -  regression with double penalty (custom3 in the paper)

## logs:

logs from NN training with input (some input with substrate smiles was ommited - one can find the reactions using reaxys' rxid)
