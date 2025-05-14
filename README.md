#  Advanced Natural Language Processing Project

## Contrastive Representation Learning for Exemplar-Guided Paraphrase Generation

### Team Members

| **Author Name** | **Roll Number** |
|-----------------|-----------------|
| Sagnick Bhar        | 2023201008             |
| Sumit Prakash        | 2023201020             |
| Aman Khurana        | 2023201017             |

---

## Table of Content
  * [Overview](#overview)
  * [Directory Structure](#directory-structure)
  * [How to run the code](#how-to-run-the-code)
  * [How to evaluate](#how-to-evaluate)

## Overview
Paraphrase generation is a fundamental task in natural language processing (NLP) with applications in text rewriting, question answering, and semantic similarity detection. This project explores the integration of Contrastive Representation Learning into exemplar-guided paraphrase generation. Traditional paraphrase generation models often struggle to balance style preservation and semantic fidelity. To address these challenges, we employ a contrastive learning framework that aligns style and content representations effectively. Our approach utilizes a Seq2Seq model with attention mechanisms, enhanced with contrastive losses for both style and content embeddings. Evaluations on the QQP-Pos and ParaNMT datasets demonstrate significant improvements in BLEU, ROUGE, and METEOR scores over baseline models. The findings suggest that the proposed method not only generates high-quality paraphrases but also provides fine-grained control over style transfer. This report details the model architecture, training methodology, experimental setup, and evaluation results, contributing a novel perspective to exemplar-guided text generation.


## Directory Structure
```plaintext
project-root/
├── datasets/
│   └── processed/                     # Processed data 
│       ├── paranmt/           
│       └── qqp-pos/            
├── dataset-text/
│   ├── quora-text/                    # Quora dataset
│   └── para-text/                     # ParaNMT dataset
├── glove/
│   └── glove.6B.300d.txt              # Glove embeddings
├── save_model/                        # Training original paper model
│   ├── our_quora/      
│   │    ├── best_seq2seq.pkl          # Trained Seq2Seq model
│   │    ├── best_stex.pkl             # Trained Style Extractor model
│   │    ├── loss.pkl                  # logs of loss per epoch
│   │    ├── ppl.pkl                   # logs of perplexity per epoch
│   │    ├──generated_paraphrases.txt  # generated paraphrase for user input
│   │    ├── exm.txt                   # generated exemplars from testing
│   │    └── trg_gen.txt               # generated test results 
│   └── our_paranmt/       
│       ├── best_seq2seq.pkl           # Trained Seq2Seq model
│       ├── best_stex.pkl              # Trained Style Extractor model
│       ├── loss.pkl                   # logs of loss per epoch
│       ├── ppl.pkl                    # logs of perplexity per epoch
│       ├──generated_paraphrases.txt   # generated paraphrase for user input
│       ├── exm.txt                    # generated exemplars from testing
│       └── trg_gen.txt                # generated test results 
├── save_model_swa/                    # Training using SWA approach
│   ├── our_quora/       
│   │    ├── swa_seq2seq.pkl           
│   │    ├── swa_stex.pkl    
│   │    ├── loss.pkl         
│   │    ├── swa_ppl.pkl       
│   │    ├──generated_paraphrases.txt  
│   │    ├── exm.txt          
│   │    └── trg_gen.txt           
│   └── our_paranmt/       
│       ├── swa_seq2seq.pkl 
│       ├── swa_stex.pkl    
│       ├── loss.pkl         
│       ├── swa_ppl.pkl      
│       ├──generated_paraphrases.txt   
│       ├── exm.txt          
│       └── trg_gen.txt        
├── train.py                           # Training script for original paper model
├── train_swa.py                       # Training script using SWA approach
├── test.py                            # Evaluation script
├── generate.py                        # Script to generate paraphrase for user input sentence
├── models.py                          # Script containing models
├── loss.py                            # Script containing loss function
├── data_preparation.py                # Script containing data preparation and loading 
├── utils.py                           # Utility functions
├── README.md                          # Project documentation
├── para_config.json                   # Contains model parameters for paranmt dataset
├── quora_config.json                  # Contains model parameters for quora dataset
└── Report.pdf                         # Project Report   
```

## Model Links
best_seq2seq.pkl (quora)= https://drive.google.com/file/d/1iNugTxqC4LLkxeEAErKGxtgYujc-qkVH/view  
best_stex.pkl (quora) = https://drive.google.com/file/d/1ztUcLfNYyLBoV0IBp4Es6Fw40jKAUHPr/view    
best_seq2seq.pkl (paranmt)= https://iiitaphyd-my.sharepoint.com/personal/aman_khurana_students_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Faman%5Fkhurana%5Fstudents%5Fiiit%5Fac%5Fin%2FDocuments%2Fproject&ga=1      
best_stex.pkl (paranmt) = https://iiitaphyd-my.sharepoint.com/personal/aman_khurana_students_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Faman%5Fkhurana%5Fstudents%5Fiiit%5Fac%5Fin%2FDocuments%2Fproject&ga=1      
swa_seq2seq.pkl (quora) = https://iiitaphyd-my.sharepoint.com/:u:/g/personal/sagnick_bhar_students_iiit_ac_in/ERN2kpZiHlBOtUxxXUlm_zIBicNTXEvXLqaJFoby_zpXUg?e=u2DiH2  
swa_stex.pkl (quora) = https://iiitaphyd-my.sharepoint.com/:u:/g/personal/sagnick_bhar_students_iiit_ac_in/EVldSNkeyoxAjPN-9rmB2EkBz__ns5tkNZaHF8fM-lYEnA?e=JM3Hp6  

## How to run the code
1. Run and download the `glove_indices.py` which downloads and saves the glove embeddings of 300 dimensions.
2. Download the dataset from [here](https://drive.google.com/drive/folders/1xkCtRnbeKPg_-0qR7j8jtzV8lfEcZGJm?usp=sharing) and put it to project directory. </br>
You can directly use preprocessed dataset`(data/: QQP-Pos, data2: ParaNMT)` </br>
Or process them `(Quora and Para)` by your own through `quora_data_preprocessing.py` and `paranmt_data_preprocessing.py` respectively.</br>
3. Follow the above directory structure 
4. To run the code: Terminal Input:
```
python train.py --dataset [quora/para] 
python train_swa.py --dataset [quora/para]  
```

## How to evaluate

1. Firstly, generate the test target sentences by running </br>
Terminal Input:
```
python evaluate.py --dataset [quora/para] --model_type [org/swa]
```
After running the command, you will find the generated target file `trg_gen.txt` and corresponding exemplar file `exm.txt` as per the directory structure</br>

## How to generate multiple paraphrases for a single user input
The ouptut would be generated under the corresponding model directory with name `generated_paraphrases.txt`
Terminal Input:
```
python generate.py --dataset [quora/para] --model_type [org/swa]
```