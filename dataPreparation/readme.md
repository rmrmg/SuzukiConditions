# Information about extraction data from reaxys and preparation input for AI

## Information about files
- SuzukiCouplingReactionRXID.txt - List of reaxysID for all extracted reactions. 
    The reaction list was subject of further filtration with parseReaxsysFile_v2.py script.
   
- parseReaxsysFile_v2.py - script which use raw-data for generating internal representation which is the use by `makeDataForKeras_v2.py` to general AI input. Before use one need to: i) get or prepare raw-data (see p. 1 in instruction below) and adjust file paths (see p. 2 below)
- makeDataForKeras_v2.py - script which prepares input for AI learning. The script use results of `parseReaxsysFile_v2.py` as a input.

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
Information about path to the files with raw data should be insert to `parseReaxsysFile_v2.py` script. you need to adjust following variables:

 - prefixheterohetero - directory where files with raw data for heteroaryl-hetoaryl Suzuki coupling are stored. Default location is 'downloadedRx/hetero-hetero/'
 - heterohetero - list of files in directory defined by `prefixheterohetero` which will be used
 - prefixarylhetero - directory where file with raw data for aryl-heteroaryl Suzuki coupling are stored. Default location is 'downloadedRx/hetero-aryl/'
 - arylhetero - list of files in directory defined by `prefixarylhetero` which will be used.
 - prefixarylaryl - directory with aryl-aryl coupling. Default location 'downloadedRx/aryl-aryl/'. This is not used for AI input preparation
 - arylaryl - list of files in directory defined by `prefixarylhetero` which will be used.
    
### 3 Convert and filter raw data
The `parseReaxsysFile_v2.py` script filter out reaction which are not Suzuki coupling (based on performed Rdkit's Reaction on subsrates). The script has following options which allow further filtration:
 - --heterohetero    include heteroaryl heteroaryl Suzuki coupling 
 - --arylhetero      include aryl heteroaryl Suzuki coupling 
 - --arylaryl        include aryl aryl Suzuki coupling 
 -  --withpatents     include also data from patents
 - --withtemponly    include only reaction with given temperature
 - --withbaseonly    include only reaction with given base
 - --withsolvonly    include only reaction with given solvent
 - --withyieldonly   include only reaction with given yield
 - --withligandonly  include only reaction with given ligand
 - --withpdonly      include only reaction with given Pd-source
 
All AI models were build on data extracted with following options: `--heterohetero --arylhetero  --withbaseonly --withsolvonly --withyieldonly --withpdonly`
 
### 4 generate input for AI
The script `makeDataForKeras_v2.py` can be used to generated most of input for AI training, depending on selected options:
 - --conditions     defined how information about reaction conditions should be encoded in AI input. Allowed values:    
   - newClasses - 'coarse-grained' classes in the paper (Table 3 entry 4)
   - oldClasses - fine classes in the paper (Table 3 entry 2)
   - embedded - embeded condition (Table 3 entry 6)
   - newClassesEmbedLig - 'coarse-grained' classes  with ligand in the paper (Table 3 entry 5)
   - oldClassesEmbedLig - fine classes with ligand in the paper (Table 3 entry 3)
 - --mode     defined how information about starting material (i.e. coupling partners) should be encoded for AI training, allowed values:
   - morgan3 - use Morgan fingerprint
   - enc - use representation from autoencoder
   - canonSmiles - use canonical smiles
   - rdkit - use rdkit descriptors as returned by rdkit.Chem.Descriptors.descList
   - morgan3rdkit - use concatenation of Morgan fingerprint and rdkit descriptors
 - --includeligand  - how information about ligands should be incorporated. Allowed values {raw,scalled} 
 - --allownotemp - when this option is used reaction without temperature will be included in AI input
 - --userawtemperature - when set raw value (i.e. Celcius degree) of temperature will be set to AI input
 - --outputMclass - in addition to main results file, generate additional file with pairs of reactions which fullfill two criteria: a) fall into the same class of base and solvent and b) difference between reactions yields is at least 25%
