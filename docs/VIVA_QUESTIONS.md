# Viva Questions and Short Answers

1. **What is text classification?**  
   Assigning a predefined category or label to a text document.

2. **What is the problem in this project?**  
   Binary sentiment classification of IMDb movie reviews into positive and negative classes.

3. **What is NLP?**  
   Natural Language Processing is a field that helps computers process and analyze human language.

4. **Why did you choose IMDb reviews?**  
   The dataset contains review text with sentiment labels, so it is suitable for supervised sentiment classification.

5. **What is preprocessing?**  
   Cleaning and standardizing raw text before feature extraction.

6. **What is TF-IDF?**  
   Term Frequency–Inverse Document Frequency converts documents into numerical vectors and weights terms based on their frequency and informativeness.

7. **Why split the data into training and testing sets?**  
   The training set is used to fit models; the test set estimates performance on unseen data.

8. **Why use stratified splitting?**  
   It maintains the class proportions in both training and test subsets.

9. **Why fit TF-IDF only on training data?**  
   To prevent information from the test set leaking into feature extraction.

10. **What does Decision Tree do?**  
    It learns a sequence of feature-based decision rules to predict a class.

11. **What does KNN do?**  
    It predicts a class using the labels of the nearest training examples.

12. **Why use cosine distance for KNN?**  
    Cosine distance compares the direction of sparse text vectors and is commonly useful for text similarity.

13. **What does SVM do?**  
    It learns a decision boundary that separates classes with a large margin.

14. **What is Linear SVM?**  
    An SVM variant that uses a linear decision boundary; it is practical for high-dimensional TF-IDF vectors.

15. **What is Naïve Bayes?**  
    A probabilistic classifier based on Bayes' theorem and a conditional-independence assumption.

16. **What is Accuracy?**  
    The fraction of all test predictions that are correct.

17. **What is Precision?**  
    Of the reviews predicted positive, the fraction that are actually positive.

18. **What is Recall?**  
    Of the actual positive reviews, the fraction correctly identified.

19. **What is F1-score?**  
    The harmonic mean of Precision and Recall.

20. **What is a confusion matrix?**  
    A table comparing actual labels with predicted labels.

21. **What are TP, TN, FP, and FN?**  
    True Positive, True Negative, False Positive, and False Negative.

22. **What is overfitting?**  
    When a model fits training data too closely and performs worse on unseen data.

23. **What is the what-if experiment?**  
    We compare SVM with C=1.0 and C=0.1 while keeping the data and features unchanged.

24. **What does SVM parameter C control?**  
    It controls the regularization trade-off. Smaller C means stronger regularization.

25. **Which model is best in your project?**  
    Answer using your actual results. Mention the metric used and support your answer with the confusion matrix.

26. **Why should you not claim one algorithm is always best?**  
    Performance depends on the dataset, preprocessing, feature representation, and parameter settings.
