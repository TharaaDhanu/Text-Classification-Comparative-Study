# Project Guide

## What you are building

A binary sentiment classifier that reads an IMDb review and predicts `positive` or `negative`. Four supervised learning algorithms are compared using one common experimental setup.

## Pipeline

1. Read the labelled CSV.
2. Validate the two expected columns and labels.
3. Create a balanced, reproducible sample.
4. Clean each review.
5. Split raw text into training and test sets (80:20, stratified).
6. Fit TF-IDF on training text only.
7. Transform the test text with the same fitted TF-IDF vectorizer.
8. Train Decision Tree, KNN, Linear SVM, and Multinomial Naïve Bayes.
9. Calculate Accuracy, Precision, Recall, and F1-score.
10. Plot the model metrics and confusion matrices.
11. Compare SVM with C=1.0 and C=0.1.

## Important concepts for explanation

- **Label:** the correct class supplied with each review.
- **Preprocessing:** cleaning text into a consistent form.
- **TF-IDF:** represents documents numerically, giving higher weight to words that are informative within a document and less common across the corpus.
- **Train/test split:** training data fits the model; test data estimates performance on unseen examples.
- **Stratification:** preserves the class ratio in both subsets.
- **Confusion matrix:** displays true negatives, false positives, false negatives, and true positives.
- **Regularization parameter C in Linear SVM:** smaller C applies stronger regularization; the actual effect must be determined from measured results.

## Interpretation checklist

After running the notebook/script, answer these questions using your own output:

1. Which algorithm has the highest Accuracy?
2. Which has the highest F1-score?
3. Does the model with highest Accuracy also have highest F1-score?
4. Which confusion matrix has the fewest false positives?
5. Which confusion matrix has the fewest false negatives?
6. How did changing SVM C affect each metric?
7. What trade-off matters most for this application?

## Reproducibility

The sample and train/test split use `random_state=42`. Keep the same random seed, data sample, preprocessing, and TF-IDF representation when comparing the four base models.
