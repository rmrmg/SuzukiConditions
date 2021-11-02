# SuzukiConditions
Here is scipts for data processing, input generation and logs from training for all models described in the paper entitled 
`Machine Learning May Simply Capture Literature Popularity Trends: The case of Suzuki-Miyaura coupling.` by Wiktor Beker, Rafał Roszak, Agnieszka Wołos, 
Nicholas H. Angello, Vandana Rathore, Martin D. Burke, and Bartosz A. Grzybowski.

# Contents
The repo consists of 4 parts placed in separated directories:
- dataPreparation/ - scripts used for preparation and filtration data. To avoid any legal doubt we are not  publish a reaction set which was extracted 
from reaxys database. One can download the dataset from reaxsys using published reaxsysID or can prepare own dataset - see readme in dataPreparation/ for details
- classification/ - scripts used for classification task with input and logs obtained from training
- regression/ - scripts used regression task and related input and logs
- gcnn_and_dan/ - scripts used for Graph Convolution Neural Network (GCNN) and Discriminative Adversarial Networks (DAN)

# How to start
1. Unfortunately we cannot publish our reaction dataset instead one need to download it manually from reaxsys or use (or generate) own dataset using information in dataPreparation/readme.md
2. When dataset is present it can be filtered and transform into for suitable for AI learining, see dataPreparation/readme.md for details
3. Once proper input is generated one can train any of tested NN models using scripts from  `classification/`, `regression/` or `gcnn_and_dan/`

# Requirments
1. Python
2. RDkit 
3. TensorFlow
4. Keras
5. Numpy
6. Scikit-learn
