# Joint Answering and Explanation for Visual Commonsense Reasoning

This repository contains data and PyTorch code for the paper [ Joint Answering and Explanation for Visual Commonsense Reasoning](https://arxiv.org/abs/2202.12626).

## Setting up and using the repo

1. This repo is based on [R2C](https://github.com/rowanz/r2c), [CCN](https://github.com/AmingWu/CCN) and [TAB-VCR](https://github.com/Deanplayerljx/tab-vcr). You should follow these links to download VCR dataset and setup the environment respectively.    
2. Download precomputed QR->A logits and representations.
* `xxxxxxxxxxxxxxxxx`
3. Reorganize data files as follow
```
data
+-- test_pickles_first_sense_match 
+-- train_pickles_first_sense_match  
+-- val_pickles_first_sense_match  
+-- vcr1images
+-- train
|   +-- train.jsonl
|   +-- attribute_features_train.h5
|   +-- bert_da_answer_train.h5
|   +-- bert_da_rationale_train.h5
|   +-- new_tag_features_train.h5
|   +-- r2c_qr2a_train.h5
|   +-- ccn_qr2a_train.h5
|   +-- tab_qr2a_train.h5
+-- val
|   +-- val.jsonl
|   +-- attribute_features_val.h5
|   +-- bert_da_answer_val.h5
|   +-- bert_da_rationale_val.h5
|   +-- new_tag_features_val.h5
|   +-- r2c_qr2a_val.h5
|   +-- ccn_qr2a_val.h5
|   +-- tab_qr2a_val.h5
+-- test
|   +-- test.jsonl
|   +-- attribute_features_test.h5
|   +-- bert_da_answer_test.h5
|   +-- bert_da_rationale_test.h5
|   +-- new_tag_features_test.h5
```
4. If you don't want to train from scratch, then download our checkpoints
```
xxxxxxxxxxxxxxxxxxx
```


## Train/Evaluate Model
1. R2C+Ours
   - To replicate our training procedure, run: 
    ```
   cd r2c_kd/models && python train_kd_infonce.py -params=kd/model_kd_infonce.json -folder={path_to_save_model_checkpoints} -plot {plot name} 
   ```
   - To evaluation the best model checkpoint(best checkpoint should be saved with name "best"), run
   ```
   cd r2c_kd/models && python eval_best_checkpoint.py -params {path_to_your_model_config} -folder {path_to_model_checkpoints}
   ```
2. CCN+Ours
   - To replicate our training procedure, run: 
    ```
   cd CCN_kd/train && python train_kd_infonce.py -params=../kd/model_kd_infonce.json -folder={path_to_save_model_checkpoints} -plot {plot name} 
   ```
   - To evaluation the best model checkpoint(best checkpoint should be saved with name "best"), run
   ```
   cd CCN_kd/train && python eval_best_checkpoint.py -params {path_to_your_model_config} -folder {path_to_model_checkpoints}
   ```
3. TAB-VCR+Ours
   - To replicate our training procedure, run: 
    ```
   cd tab-vcr-master_kd/models && python my_train_kd_infonce.py -params=kd/default_kd_infonce.json -folder={path_to_save_model_checkpoints} -plot {plot name} 
   ```
   - To evaluation the best model checkpoint(best checkpoint should be saved with name "best"), run
   ```
   cd tab-vcr-master_kd/models && python my_eval_best_checkpoint.py -params {path_to_your_model_config} -folder {path_to_model_checkpoints}
   ```
## Bibtex
xxxxxxxxxxxxxxxx

