Hwere are files and scripts to extract Suzuki coupling reaction from USPTO data


- parseUSPTOdata.py  - this file extract Suzuki reaction from USPTO by default it looking for two files
    '1976_Sep2016_USPTOgrants_smiles.rsmi' '2001_Sep2016_USPTOapplications_smiles.rsmi' and products six files:
    with following suffix: 'raw_homo.txt' 'raw_hete.txt' 'parsed_homo.txt' 'parsed_het.txt', 'clear_homo.txt' 'clear_het.txt'

- makeReactionFromParsed.py  

- basescanon.smi  
- liganscanon.smi  
- solventscanon.smi  
- uspto_exclude_list.smi  
- uspto_replace_list.csv