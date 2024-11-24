
# Document-Level Text Simplification

This repository contains code and data from my thesis: **Document-Level Text Simplification**, accessible [here](https://pergamos.lib.uoa.gr/uoa/dl/object/3338660/file.pdf). This project focuses on developing and evaluating a deletion-driven approach for automatic document-level text simplification, leveraging advanced sentence embedding techniques and classifiers.

---

## Abstract

Text simplification is a valuable technique for improving knowledge dissemination and enhancing accessibility. This thesis introduces a **sentence deletion-based approach** to simplify documents by reducing linguistic complexity while preserving the core meaning. Significant contributions include:
1. **Unsupervised Sentence Alignment**: An automatic pipeline identifies simplification operations, including sentence deletion, insertion, merging, splitting, and paraphrasing.
2. **Large-Scale Dataset Creation**: Over 100,000 sentences were automatically aligned and labeled from the "D-Wikipedia" dataset for training and evaluation.
3. **Deletion-Driven Simplification**: Two classifiers (SVM-based and BERT-based) were trained to predict sentence deletions.
4. **Simplified Text Generation**: Predicted deletions were used to generate simplified texts, evaluated against six baselines and an oracle model.
5. **Performance Analysis**: The BERT-based classifier outperformed the SVM-based model in D-SARI and other evaluation metrics, affirming the effectiveness of deletion-based ADTS.

---

## Contribution of the Study

This thesis makes significant contributions to document-level text simplification:
- **Alignment Pipeline**: Developed an unsupervised method for aligning sentences between source and simplified texts, capturing key operations (e.g., deletion, insertion).
- **Simplification Classifiers**: Trained and evaluated SVM and BERT-based classifiers to predict sentence deletions.
- **Evaluation Metrics**: Highlighted the limitations of current metrics (D-SARI, BLEU) and the need for improved evaluation measures for ADTS.
- **Language Models**: Demonstrated the impact of large language models like BERT in enhancing simplification performance.

---

## Overview

Document-level text simplification improves readability and comprehension by reducing complexity. This project addresses ADTS by:
1. Aligning sentences between source and simplified text pairs.
2. Identifying simplification operations (deletion, insertion, splitting, merging, paraphrasing).
3. Training classifiers to predict deletions for generating simplified documents.

Key findings show that **sentence deletion is a foundational operation** for generating simpler texts, with BERT-based classifiers achieving superior performance.

---

## Repository Structure

- **`sentalign.py`**: Implements sentence alignment using Sentence Transformers and algorithms (Itermax, Match) based on SimAlign.
- **`tagger.py`**: Labels sentences as deleted/retained for training deletion classifiers.
- **`metrics.py`**: Provides evaluation metrics (D-SARI, ROUGE, BLEU, FKGL) for simplification.
- **`split_dataset.py`**: Splits data into training, validation, and test sets using stratified splitting.
- **`baseline_initial.py`**: Implements the SVM-based classifier for sentence deletion prediction.
- **`evaluate.py`**: Evaluates simplification models and generates performance metrics.
- **`features.py`**: Extracts sentence-level features (e.g., readability scores, TF-ISF, sentence position).
- **`Fine-tune BERT for Text Classification.ipynb`**: Notebook for fine-tuning BERT for sentence deletion classification.
- **Sample Data**:
  - **`sample_200_source`** and **`sample_200_target`**: Example source and simplified text pairs for alignment.
- **Training Data**:
  - **`train.csv`**, **`val_data.csv`**, **`test_data.csv`**: Preprocessed data splits for training and evaluation.

---

## Workflow

### 1. Sentence Alignment
- Use `sentalign.py` to align sentences between source and simplified texts. Outputs alignments capturing operations like deletion and merging.

### 2. Data Labeling
- Run `tagger.py` to label sentences for training classifiers (e.g., deletion vs. retention).

### 3. Model Training
- **SVM-Based Classifier**: Implemented in `baseline_initial.py`.
- **BERT-Based Classifier**: Implemented in `Fine-tune BERT for Text Classification.ipynb`.

### 4. Evaluation
- Evaluate generated simplified texts using `evaluate.py`:
  - **Metrics**: D-SARI, ROUGE, BLEU, FKGL.

---

## Results

### Sentence Alignment
- **Best Aligner**: "all-mpnet-base-v2 with Itermax" achieved micro P/R/F scores of 90.0/82.62/86.15.

### Simplification
- **SVM Classifier**: D-SARI: 21.21 | P/R/F: 70.71/70.88/70.78.
- **BERT Classifier**: D-SARI: 25.64 | P/R/F: 70.38/70.07/70.20.

---

## Limitations

1. **Operation Scope**: This work focuses on deletion-based simplification and does not extensively explore other operations (e.g., insertion, merging).
2. **Evaluation Metrics**: Existing metrics (D-SARI, BLEU) have limitations for assessing document-level simplification. Improved evaluation methods are needed.
3. **Language Dependency**: Findings are specific to English and may not generalize to other languages without adaptation.

---

## Future Work

1. Expand the framework to include other simplification operations (e.g., paraphrasing, merging).
2. Develop new evaluation metrics tailored for document-level simplification.
3. Extend the approach to support multiple languages.

---

## Requirements

Install the necessary dependencies:
```bash
pip install pandas numpy scikit-learn transformers sentence-transformers nltk textstat rouge-score
```

