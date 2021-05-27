# NT5?! Training T5 to Perform Numerical Reasoning

**Authors**: Peng-Jian Yang<sup>a</sup>, Ying Ting Chen<sup>a</sup>, Yuechan Chen<sup>a </sup>  
**Advisor**: Daniel Cer<sup>a, b</sup>   
<sup>a</sup>University of California Berkeley, <sup>b</sup>Google Research   

*NT5?! Training T5 to Perform Numerical Reasoning* is a NLP research project on training T5 to perform NRoT (numerical reasoning over text). Latest version of the paper [can be reviewed on ArXiv](https://arxiv.org/abs/2104.07307). All source codes and two fully trained NT5 models (**RC Experiment 1**, our best performing model, and **Validation Experiment 2**, our second best performing model) are included in the repository.  

## Abstract

Numerical reasoning over text (NRoT) presents unique challenges that are not well addressed by existing pre-training objectives in NLP. We explore five sequential training schedules that adapt a pre-trained T5 model for NRoT. Our final model adapted from T5 but further pre-trained on three datasets designed to strengthen skills necessary for NRoT and general reading comprehension before being fine-tuned on Discrete Reasoning over Text (DROP) dataset. We show that our training improves DROP’s adjusted F1 performance (a numeracy-focused score) **from 45.90 to 70.83**. Our model outperforms the best model in the original DROP paper (47.01), and closes in on GenBERT (72.4), a custom BERT-Base model with significantly more parameters.

## NRoT Challenges   

NRoT in NLP is unique in that answers require **numerical reasoning** in addition to the traditional NLP task, **reading comprehension (RC)**. Additionally, answers can demand the model to be both **generative** and **discriminative**, as demonstrated by the two examples extracted from DROP, our gold dataset:  

<p align="center"><img src="https://www.dropbox.com/s/abbjlaalmn00ozi/num_example.jpg?raw=1" width="800"/></p>

The answer for the **first question** is an extraction from the passage, and requires the model to compute the probability distribution across all words in the passage. In particular, our chosen model requires the following **three NLP skills in sequence**:  

<p align="center"><img src="https://www.dropbox.com/s/lq5ldksof5x4vu5/nrot.jpg?raw=1" width="650"/></p>  

The answer for the **second question**, on the other hand, cannot be extracted from either the passage or question. We need a **generative language model** to generate the string, "4300000." 

<p align="center"><img src="https://www.dropbox.com/s/xg3bhfoffw95lh1/nrot2.jpg?raw=1" width="650"/></p>  

Note that many NRoT models, including the current state of the art for solving DROP, **only** generates the mathematical equations required to calculate the final answer as the output. Our research aims to take it **one step further**: Our final model internalizes the equation, perform the calculation, and directly generate the final numerical answer, 4,300,000, as the output.  

## Datasets

- A total of **6 datasets** are explored during training. The splits and sizes for each dataset are summarized by the diagram below. 

  <p align="center"><img src="https://www.dropbox.com/s/w01jfmt0s15zco5/datasets.png?raw=1" width="650" /></p>

  - **DROP** (Discrete Reasoning Over Paragraphs), introduced by AllenNLP in 2019, includes 96k examples in a "Q&A with context" format similar to SQuAD. The benchmark includes four distinct types of questions, all of which require NRoT skills to solve. **DROP Class** is exactly the same as DROP, but with the labels changed to the four classes of questions found in DROP : numeric, date, single span, and multiple spans. The goal of DROP Class is to help T5 learn to classify the four types of questions that require different skillsets to solve in DROP.

  - **Synthetic Data** consists of two datasets: The Numeric dataset (NUM) with near 1M synthetically generated questions on seven types of numerical skills (e.g. addition, sorting, comparison, etc.). The Textual dataset (TXT) builds on NUM, and includes 2M+ synthetically generated examples in formats similar to DROP's Q&As. 

  - **SQuAD v1.1**, a benchmark dataset by Stanford with an emphasis on RC through Q&As, is included in training to strengthen the model's general RC capability. 

  - Unfortunately, we are unable to complete our multitask training with **C4EN** (used a part of T5's pre-training) due to limited resources, but we hypothesize that the inclusion of which would lead to an improved performance.  

## Evaluation

We employ two evaluation metrics: **Exact-Match (EM)**, and an **adjusted F1** (macro-averaged, adjusted for numerical performance). EM uses that same criteria as SQuAD. The adjusted F1 has additional logic that invalidates all matching material within an answer when there is a numeric mismatch. In other words, the prediction receives an **F1 score of 0** if it gets the number wrong. In addition, F1 is computed using macro-averaging over individual answers. In the presence of **multiple ground truths**, both EM and F1 will take a max over all computed scores.  

## Model Selection 

At the time of research, BERT with self-attention is becoming increasingly popular across a wide variety of NLP tasks. However, inspired by **WT5** ([Sharan et al., 2020](https://arxiv.org/abs/2004.14546) ) and **GenBERT** ([Geva et al., 2020](https://arxiv.org/abs/2004.04487)), we choose **T5** as our model specifically for its following strengths:

* **Multitasking, enabled by T5's Text-to-Text Framework** :
  * T5 can concurrently train on multiple datasets with distinct label formats. 
  * One single T5 model can be fine-tuned against multiple objectives and perform different types of predictions. This is a **strong contrast to BERT** and an important feature required by DROP, our gold dataset, as explained in the NRoT challenges section.      
* **Strong RC from transfer learning**: 
  * T5 is Google's attempt to take transfer learning to its limit across a wide verity of NLP tasks. It is pre-trained on Colossal Clean Crawled Corpus (C4).
  * A short description of T5 can be found on the [Google AI Blog](https://tinyurl.com/yg6wsf38) and T5's [original paper](https://arxiv.org/abs/1910.10683). 

- **Parsimonious architecture & training process**:
  - T5 allows us to complete the entire training schedule using its **out-of-box architecture**. BERT, on the other hand, requires additional feedforward neural networks for fine-tuning. 
  - T5's multitasking allows us to fine-tune **one single model** for all the different NLP tasks demanded by DROP. In contrast, BERT requires multiple different models to solve DROP.  
  - We hypothesize that T5's pre-training and encoder-decoder architectures would lead to a performance comparable to BERT but with a much **smaller model scale**.

## Training Methodology

The parsimony of T5 allows us to **focus on refining our training methods** instead of the model architecture. Our training involves a series of experiments using both sequential and multitask trainings. The full schedule is summarized by the diagram below and a detailed description can be found in the paper.   

<p align="center"><img src="https://www.dropbox.com/s/ujq8dc229nl79bh/schedule.jpeg?raw=1" width="800" /></p>



## Results

Our model using T5-Small (the smallest scale of T5) achieves an adjusted F1 performance of **70.83**. This is a considerable improvement over the performance achieved by the model proposed in the original DROP paper (47.01). Our model also closes in on GenBERT (72.4), a custom BERT-Base model pre-trained on our same synthetic data. In addition, our model is a lot more parsimonious: GenBERT's architecture includes 5 additional feedforward neural networks on top of the BERT-Base encoder and comes with **significantly more weights** (110 million from BERT-Base + additional weights from the 5 neural networks vs. 60 million from our T5-Small).    

## Repository 
- [`./nt5_multitask_training.ipynb`](./nt5_multitask_training.ipynb) is the notebook with all source codes for modeling and training.
- [`./models`](./models) includes two fully trained models in h5 format: our best (**RC Experiment 1**) and second best (**Validation Experiment 2**) performing models. 
- [`./error_analysis`](./error_analysis) includes the evaluation reports for the two models included in `./models`.
- `./data` includes the datasets we used for training NT5.
- `./tfrec` includes the source codes used to generate the TFRecord files used for data streaming during training.

