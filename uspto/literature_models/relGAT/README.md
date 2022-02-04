Changed files from [RelGAT](https://github.com/slryou41/reaction-gcnn) repository

- train.py
- predict.py

added file:
- calc_top_k_uspto.py


# Pipeline:

## TRAINING:

for FOLD in {0..4}:

python reaction-gcnn/train.py -r 0.999 --method relgat  --epoch 50 --data-name suzuki_rr_ours_uspto -o rr_gat_result_e50_f$i --fold $FOLD --gpu 0 -b 16


## EVALUATION:

for FOLD in {0..4}, epoch in {1..50}: 

python reaction-gcnn/predict.py --method $method --data-name suzuki_rr_ours_uspto -i rr_gat_result_e50_f$FOLD --fold $FOLD --gpu $GPU --load-modelname rr_gat_result_e50_f$FOLD/model_epoch-$i --app _e$i

## Collecting Results:

python calc_top_k_uspto.py --directory rr_gat_result_e50 --cv --option suzuki_rr_ours_uspto --out rr_relgat_ml10_e50_cv_result.csv

