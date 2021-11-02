## Information about extraction data from reaxys and preparation input for AI

# Information about files
- SuzukiCouplingReactionRXID.txt - List of reaxysID for all extracted reactions. 
    The reaction list was subject of further filtration with parseReaxsysFile_v2.py script.
   
- parseReaxsysFile_v2.py - script for
- makeDataForKeras_v2.py

# Instruction for data preparation
1. download data from reaxsys (or other prefered source) input data should be in text format with value separated by tab and first line is a header. 
 Such format is call tsv (tab-separated-values) or csv with tab delimiter.

2.