# Information about extraction data from reaxys and preparation input for AI

## Information about files
- SuzukiCouplingReactionRXID.txt - List of reaxysID for all extracted reactions. 
    The reaction list was subject of further filtration with parseReaxsysFile_v2.py script.
   
- parseReaxsysFile_v2.py - script for
- makeDataForKeras_v2.py

## Instruction for data preparation
### 1. Download or preaprare raw data 
We download raw data from reaxsys however you can use any source of literature reports. The raw data should be saved in text format with value separated by tab and the first line should be a header. Such file format is call tsv (tab-separated-values) or csv with tab delimiter. Original reaxys format consists of 37 fields, however only following fields are obligatory in our pipeline:
 - 'Solvent (Reaction Details)' - information about solvents will be read from this column only. 
 - 'Temperature (Reaction Details) [C]' - reaction temperature in Celcius degree as number.
 - 'Time (Reaction Details) [h]' - reaction time in hours. This field is not included in AI input however it is required but can be blank.
 - 'Yield (numerical)' - reaction yield as number in range 0-100.
 - 'Other Conditions' - further information about reaction conditions, required but can be blank.
 - 'Reactant' - information about base is reading from fields 'Reactant' and 'Reagent' only.
 - 'Reagent' - see above
 - 'References' - source of report. Reaction report is clasified as patent when word 'patent' is in the field otherwise is clasified as 'article'

 Reaxys format is not very strict or well defined therefore for some informations (like catalysts or ligand) we search in whole line expect defined below fields:
    Following fields are ignored when looking for catalyst:
    - 'Fulltext of reaction
    - 'Example title' 
    - 'rxInfo'

 Following fields are ignored when looking for ligands:
    - 'Fulltext of reaction' 
    - 'References', 
    - 'Product'
    - 'rxInfo'

### 2. Add path to raw files
Information about path to the files with raw data should be insert to parseReaxsysFile_v2.py script. you need to adjust following variables:

 - prefixheterohetero - directory where files with raw data for heteroaryl-hetoaryl Suzuki coupling are stored. Default location is 'downloadedRx/hetero-hetero/'
 - heterohetero - list of files in directory defined by `prefixheterohetero` which will be used
 - prefixarylhetero - directory where file with raw data for aryl-heteroaryl Suzuki coupling are stored. Default location is 'downloadedRx/hetero-aryl/'
 - arylhetero - list of files in directory defined by `prefixarylhetero` which will be used.
 - prefixarylaryl - directory with aryl-aryl coupling. Default location 'downloadedRx/aryl-aryl/'. This is not used for AI input preparation
 - arylaryl - list of files in directory defined by `prefixarylhetero` which will be used.
    
### 3 Convert and filter raw data

### 4 generate input for AI

