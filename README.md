# ML-NaiveBayes_learning

			


Naive Bayes:
It is a probabilistic machine learning model which is utilized for classification tasks called Naïve Bayes Classifier.
Task-1
Platform: Jupyter Notebook
 
Here I have imported the required libraries for the tasks
Pandas: By definition. It can be used to perform data manipulation and analysis
NumPy: It is for Array concepts and has functions to work in the field of Mathematics such as linear algebra and Matrices etc. 
Matplotlib: which is used as a Visualize the given dataset. 
Seaborn: which is used to Visualize the random distributions (Statistical Graphs)
 
Here, defined the file and created the two lists namely: Docs and labels where utf-8 is the file code
 
Here, reading the file and printed the docs and labels where labels are separated from documents.
 
Providing a dataframe for the docs and lables for the analysis
 
Here, we are checking the No of rows and columns in a data
 
In this step, removed the punctuation marks from documents and labeled them 1 for the pos and 0 for the neg which are positive and negative and this is called as Binary transformation.
  
From the above label distribution, we can see there is minor difference between the pos and neg which are replaced with 0 & 1.
 
Imported the necessary libraries like train_test_split from the sklearn.model_selection and splitting the data into 80 % for training and 20 % for testing.
 
In this step, built the pipeline that contains the transformers which are Countvectorizer, tfidftransformer and multinomialNB
tfidftransformer: With the help of the "Term Frequency Transformer," words that often appear in training data and are consequently less helpful for the estimator than keywords that appear in a smaller percentage of text samples have less of an impact.
multinomialNB: Another helpful Naive Bayes classifier is this one. It presumptively uses a straightforward multinomial distribution as the source of the features.
CountVector and TdidfTransformer turn the input text of strings into numerical entries, which are then passed as inputs to MultinomialNB.
 
To quantify the model, we use the confusion matrix, and the results are printed
 
Retrieved Source:
https://www.simplilearn.com/tutorials/machine-learning-tutorial/confusion-matrix-machine-learning
 
From the visualizing the confusion matrix, we can calculate the accuracy of the model by formula where the TP=1030, TN = 982, FP = 195 and FN = 176.
The accuracy of the model is 84.43 %.
 
An evaluation statistic for machine learning performance is a classification report. It is a statistic for measuring the effectiveness of machine learning that shows the precision, recall, F1 Score, and support score of your developed classification model
 
From the above results, we can see the cross-validation scores and the accuracy for the cv which is 0.827 +/- 0.009 through the No of folds (CV=10).



Error Analysis:
 


 
 
 

In the last step, applied test data log probabilities for error analysis and test_data_err data frame, we are concatenating the outcomes of log probabilities. We initialize the error diff by taking the absolute difference between the pos and neg probability result values. To sort the values by error diff column in ascending order, we are now querying the data when label does not equal prediction.
Conclusion:
From the above results, we can see from the data above that out of a total of 2383 records, 371 were incorrectly predicted. In proportion, the model incorrectly labelled 15.56% of the records, giving us a model accuracy of 84.431%.


