# NT5?! Training T5 to Perform Numerical Reasoning

Tim (Ying Ting) Chen, Lester (Peng-Jian) Yang, Sonya Chen, Daniel Cer

Abstract
Numerical reasoning over text (NRoT) presents unique challenges that are not well addressed by existing pre-training objectives. We explore five sequential training schedules that adapt a pre-trained T5 model for NRoT. Our final model adapted from T5 but further pre-trained on three datasets designed to strengthen skills necessary for NRoT and general reading comprehension before being fine-tuned on Discrete Reasoning over Text (DROP) dataset. We show that our training improves DROP’s adjusted F1 performance (a numeracy-focused score) from 45.90 to 70.83. Our model outperforms the best model in the original DROP paper (47.01), and closes in on GenBERT (72.4), a custom BERT-Base model with significantly more parameters.


Repository Description
The repository includes source codes used to train NT5, a T5 trained to perform numerical reasoning using the DROP dataset. Files included are:
- `data` includes the datasets we used for training NT5.
- `error_analysis` includes a sample of the reports used to evaluate NT5’s performance.
- `models` includes a sample NT5 model.
- `tfrec` includes the source codes used to generate the TFRecord files used for data streaming during training.
