# lhs712_task2

This repository contains our final report and each of our individual code files for SMM4H Task 2: Automatic Classification of Adverse Events. Details are included below for how to run each of our code files (e.g., packages required, location of data files). Please reach out to us if any additional questions arise!


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
