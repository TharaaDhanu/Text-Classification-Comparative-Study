# Project Report: Comparative Study of Machine Learning Algorithms for Text Classification

> Complete the bracketed result fields only after running your own experiment. Do not invent metrics.

## 1. Abstract
This project compares Decision Tree, K-Nearest Neighbours (KNN), Linear Support Vector Machine (SVM), and Multinomial Naive Bayes for binary sentiment classification of IMDb movie reviews. Text is cleaned and converted into TF-IDF features. All classifiers use the same stratified training/testing split and feature representation. Accuracy, Precision, Recall, F1-score and confusion matrices are used for evaluation. A what-if experiment compares two SVM regularization settings, C=1.0 and C=0.1.

## 2. Introduction
Text classification assigns predefined labels to textual documents. Sentiment analysis is a common text-classification task that identifies whether a review expresses a positive or negative opinion. This project compares four classical supervised learning algorithms under a shared experimental protocol.

## 3. Problem statement
To compare Decision Tree, KNN, SVM and Naive Bayes for classifying IMDb movie reviews as positive or negative using TF-IDF features and a consistent evaluation procedure.

## 4. Objectives
- Select a labelled text dataset and define its classes.
- Preprocess review text.
- Convert text into numerical TF-IDF features.
- Train four classifiers using the same data split and features.
- Evaluate using Accuracy, Precision, Recall and F1-score.
- Analyze confusion matrices and compare results.
- Conduct a parameter-based what-if experiment.
- Recommend a model based on the actual results.

## 5. Dataset
Dataset: IMDb 50K Movie Reviews.
Input: review text.
Target: sentiment.
Classes: positive and negative.
Experimental sample: 5,000 reviews per class (10,000 total), sampled with random_state=42.

## 6. Preprocessing and feature extraction
The experiment removes missing rows, normalizes sentiment labels, selects a balanced sample, converts reviews to lowercase, removes HTML tags and non-letter characters, and normalizes whitespace. `TfidfVectorizer` uses at most 10,000 features, English stop-word removal and unigrams/bigrams. TF-IDF is fitted on training text only and then applied to test text.

## 7. Experimental protocol
The dataset is divided into 80% training and 20% testing using a stratified split with random_state=42. All four algorithms use exactly the same split and TF-IDF features:
- Decision Tree (`random_state=42`)
- KNN (`n_neighbors=5`, cosine distance)
- Linear SVM (`C=1.0`)
- Multinomial Naive Bayes (default smoothing)

## 8. Evaluation metrics
Accuracy = (TP+TN)/(TP+TN+FP+FN)
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1-score = 2 × Precision × Recall/(Precision+Recall)

Positive is treated as the positive class for Precision, Recall and F1-score.

## 9. Results
Paste your generated `results/algorithm_comparison.csv` values here.

| Algorithm | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Decision Tree | [run result] | [run result] | [run result] | [run result] |
| KNN | [run result] | [run result] | [run result] | [run result] |
| SVM | [run result] | [run result] | [run result] | [run result] |
| Naive Bayes | [run result] | [run result] | [run result] | [run result] |

Insert `performance_comparison.png` and `confusion_matrices.png`. Discuss TP, TN, FP and FN from each confusion matrix, identifying which error type is more frequent for each model.

## 10. What-if experiment
Only the SVM regularization parameter is changed:
- Original: C=1.0
- Modified: C=0.1

Paste the actual metrics from `results/svm_what_if_comparison.csv`.

| Metric | SVM C=1.0 | SVM C=0.1 |
|---|---:|---:|
| Accuracy | [run result] | [run result] |
| Precision | [run result] | [run result] |
| Recall | [run result] | [run result] |
| F1-score | [run result] | [run result] |

Explain the observed difference using your measured values. A smaller C applies stronger regularization; its effect on generalization must be determined from the experiment rather than assumed.

## 11. Discussion
Compare the measured results. Decision Trees learn rule-based splits but can overfit. KNN predicts using nearby examples and may be expensive at prediction time. Linear SVM is designed to work with high-dimensional feature spaces. Naive Bayes estimates class probabilities under a conditional-independence assumption. Explain which of these characteristics is consistent with the observed metrics and confusion matrices.

## 12. Conclusion and recommendation
In this run, [algorithm] achieved the highest [metric] with a value of [actual value]. Based on the combined metrics and confusion matrix, [algorithm] is recommended for this dataset and experimental setup. This conclusion is limited to the chosen sample, preprocessing, TF-IDF settings and model parameters.

## 13. References
1. IMDb 50K Movie Reviews dataset, Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2. Scikit-learn documentation: https://scikit-learn.org/stable/
3. Scikit-learn TF-IDF documentation: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
