# lhs712_task2

This repository contains our final report and each of our individual code files for SMM4H Task 2: Automatic Classification of Adverse Events. Details are included below on how to run each of our code files (e.g., packages required, location of data files). Please reach out to us if any additional questions arise!


Emmi Carr:

-Packages required: pandas, numpy, nltk, gensim, re, string, emoji, keras, tensorflow, sklearn

-Data files must be located in the same file as the code file. Additionally, if you would like to run the code to manually create the word embeddings, you must also have the task 1 training & validation data in the same location. Conversely, the file paths in the code can also be changed to where the data is stored on your device.

-Additional Notes: please be patient when running the file as some of the code can take ahwile to run! I uploaded my code in .ipynb and .html formats.


Siddharth Madapoosi:

-Packages required: tidyverse, tidytext, caret, stringr, kernlab, pROC, and tm

-RxTerms data should remain in its own folder

-File paths need to be updated based on where you have the data stored

-TF_Linear_SVM_LogitBoost_RxTerms.R has a preprocessing step that replaces brand-name drug references in tweets with “drug” according to RxTerms and runs linear SVM and logitboost, while TF_Linear_SVM_LogitBoost.R does not have the RxTerms drug name replacement step (the latter script was used to generate models 6, 8, and 10 in Table 1)

-Both scripts will save the final models as RDS files, create a table comparing the precision, recall, F1, and accuracy at a term frequency of 3 and 5, and a graph comparing F1 scores at the corresponding number of features and take approximately 5-10 minutes to run


Tasha Torchon:

-The code should be in the same folder as the SMM4H data, which should have the original names: “task2_en_training.tsv” and “task2_en_validation.tsv”.

-In the command terminal output, “Trained with Original Data” refers to models that were trained with “task2_en_training.tsv” as given by SMM4H. “Trained with Clean Data” refers to the results from models trained with the preprocessed data. 

-The code should take 5-10 minutes to run.


Angel Ka Yan Chu:

Getting Started:

-R version 3.6.2 was being used throughout the analysis. File paths should be updated and use setwd to specify the path to the desired folder and to change the working directory.

Prerequisites and Installing:

-A list of required packages are listed in the beginning of the R code. Some essential ones for model building are: quanteda, e1071, caret, and modelr. For the purpose of visualization, one would install ggplot2 and wordcloud2. Install dplyr, tidytext, and tokenizer for data cleaning steps. 
-The code can be broken down into six major sections: 

(1) Basic cleaning, pre-processing and filtering

(2) Sentimental analysis with lexicon bing and visualization

(3) SVM model training

(4) Model evaluation for SVM model

(5) Random Forest training

(6) Model evaluation for Random Forest

Running the models:

-Depending on how many features and/or texts there are in the training dataset, the SVM model might take approximately 20 minutes. The Random Forest model takes more time to train, about 30 minutes. 

Running the metrics:

-Functions that are used to compute accuracy, precision, recall, F1-score, and confusion matrix are included. 

