# Identifying Constructive Conflict in Online Discussions through Controversial yet Toxicity Resilient Posts 

This repo contains code to reproduce the results in the paper. 

## T model:
N/A, we use openAI-omni-moderation for the results reported in the paper

## TA model training:
- Script: `scripts/hyperparam_fine_tune_distilbert_ta.py`
- Data: accessible through Zenodo. Data consists of 2 columns: submission-id and TA-score. To use this data, download files from <link> and put it in `data/`. Researchers should hydrate the text data by themselves by merging this dataset with the original data provided by Reddit.

## C model training:
- Script: `scripts/hyperparam_fine_tune_distilbert_c.py`
- Data: We used the data curated by [Sznajder et al. 2019](https://arxiv.org/abs/1908.07491). To use this data, download files from [IBM Project Debater Dataset III](https://research.ibm.com/haifa/dept/vst/debating_data.shtml#:~:text=Go%20to%20download%20%E2%86%93-,Concept,-Controversiality) and put it in `data/IBM_Concept_Controversiality`

Note: all model training scripts use the helper functions from: `scripts/train_utils.py`

## Model validation
The annotation results and related manual annotation files can be found in the following directories:
- Toxicity model (note that this is not to evaluate TA model but to choose the best toxicity model that would help us create a training data for TA model): `validation/toxicity_manual_eval` 
- C model: `validation/controversiality_manual_eval`

## Analysis
- Baseline model for TA (submission toxicity using linear regression): `results/baseline_model.ipynb`
- Produce TA, C score and BERTopic: `results/run_trained_models_and_BERTopic_on_test_data.ipynb`
- Interplay between TA and C (Fig.1): `results/TA_vs_C.ipynb`
- Topic analysis (Fig.1): `results/topic_analysis_on_quadrants.ipynb`
- Extract linguistic features: `results/linguistic_features.ipynb`