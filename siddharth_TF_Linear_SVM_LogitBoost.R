options(warn=-1)
suppressMessages(library(tidyverse))
suppressMessages(library(tidytext))
suppressMessages(library(caret))
suppressMessages(library(tm))
suppressMessages(library(stringr))
suppressMessages(library(kernlab))
suppressMessages(library(pROC))

#read data
training_data <- read.csv('~/Desktop/LHS_712/Final_Project/task2_en_training.tsv', sep = '\t', header = TRUE, stringsAsFactors = FALSE)

#read database of terms
rxterms <- read.table('~/Desktop/LHS_712/Final_Project/RxTerms202003/RxTerms202003.txt', sep = '|', header = TRUE, fill = TRUE)

#get brand names to conduct medical dictionary
#brand_names <- tolower(unique(rxterms$BRAND_NAME)[-2]) #get list of unique brand names in Rxterms and remove the blank one
#brand_names <- tibble(brand_names)
#colnames(brand_names) <- "word" #rename bran name column to word to merge with training data

set.seed(1234)
#training_data <- training_data[sample(rownames(training_data), size = 5000), ] #sample n random tweets to check computational time

#number of tweets that a word must appear in to be a feature for model
num_tweets_min <- c(10, 5, 3) #tune threshold number of tweets a word must appear in to see impact on F1

#tokenize tweets in training data and remove stop words
training_data_tokenized_no_stops <- map_df(1:2, ~ unnest_tokens(training_data, 
                                           						word, 
                                           						tweet, 
                                           						token = "tweets")) %>%
                                           						#n = .x)) %>% #tokenize
									anti_join(stop_words, by = "word") #remove stop words

#count number of occurrences of each word in each tweet
data_counts <- training_data_tokenized_no_stops %>%
				count(tweet_id, word, sort = TRUE) #count number of occurrencs of each term in each tweet and sort

num_features_list <- list(NULL)

#create table of model statistics for linear SVM and logitboost models
#F1
final_linear_f1_list <- list(NULL)
final_logitboost_f1_list <- list(NULL)
#Precision
final_linear_precision_list <- list(NULL)
final_logitboost_precision_list <- list(NULL)
#Recall
final_linear_recall_list <- list(NULL)
final_logitboost_recall_list <- list(NULL)
#Accuracy
final_linear_accuracy_list <- list(NULL)
final_logitboost_accuracy_list <- list(NULL)

for (i in 1:length(num_tweets_min)) {
	print("*****************")
	print(paste("Starting: ", num_tweets_min[i], sep = ""))
	#look only at words that appeared in at least num_tweets_min counts
	words_min <- data_counts %>%
	            group_by(word) %>%
	            summarise(n = n()) %>% 
	            filter(n >= num_tweets_min[i]) %>%
	            select(word)
	num_features_list[i] <- nrow(words_min) #record number of features 

	#print number of words removed by threshold
	words_no_threshold <- data_counts %>%
	            			group_by(word) %>%
	            			summarise(n = n()) %>% 
	            			filter(n >= 1) %>%
	            			select(word)
	print(paste(nrow(words_no_threshold) - nrow(words_min), 
				" of ",
				nrow(words_no_threshold),
				" words were removed by thresholding at ", 
				num_tweets_min[i], 
				sep = ""))

	#right join the words_min data frame to data frame and cast it to a document term matrix
	data_dtm <- data_counts %>%
	            right_join(words_min, by = "word") %>% 
	            bind_tf_idf(word, tweet_id, n) %>% 
	            cast_dtm(tweet_id, word, tf_idf)

	#create intermediate data frame
	meta <- tibble(tweet_id = as.numeric(dimnames(data_dtm)[[1]])) %>% 
	               left_join(training_data[!duplicated(training_data$tweet_id), ], 
	                         by = "tweet_id")

	#create training set and validation set
	set.seed(1234)
	trainIndex <- createDataPartition(na.omit(meta$class), p = 0.8, list = FALSE, times = 1)
	data_df_train <- data_dtm[trainIndex, ] %>% as.matrix() %>% as.data.frame()
	data_df_validation <- data_dtm[-trainIndex, ] %>% as.matrix() %>% as.data.frame()

	#missing rows
	missing_labels_training <- which(is.na(meta$class[trainIndex]))
	data_df_train <- data_df_train[-missing_labels_training, ]

	response_train <- meta$class[trainIndex]
	response_train <- response_train[-missing_labels_training]

	#look at missing tweets
	missing_tweet_count <- training_data %>% anti_join(meta, by = "tweet_id") %>% nrow()
	missing_tweet_proportion <- missing_tweet_count/nrow(data_df_train) * 100

	#***********************************************************************************************************#
	#***Initial Models***#
	print("Starting Initial Models")
	start.time <- Sys.time() #record start time
	set.seed(1234)

	trctrl <- trainControl(method = "none") #no cross-validation

	#LogitBoost Model as Baseline#
	print("Starting LogitBoost")
	logitboost_model <- train(x = data_df_train,
                          y = as.factor(response_train),
                          method = "LogitBoost",
                          trControl = trctrl)
	logitboost_predict <- predict(logitboost_model, newdata = data_df_validation)
	logitboost_confusion_matrix <- confusionMatrix(logitboost_predict, 
                                               as.factor(meta[-trainIndex, ]$class), 
                                               positive = '1', 
                                               mode = "prec_recall")
	#get model statistics
	f1_score_logitboost <- logitboost_confusion_matrix[[4]][[7]]
	precision_logitboost <- logitboost_confusion_matrix[[4]][[5]]
	recall_logitboost <- logitboost_confusion_matrix[[4]][[6]]
	accuracy_logitboost <- logitboost_confusion_matrix[[3]][[1]]

	#replace F1 score with 0 if NA due to poor performance
	if (is.na(f1_score_logitboost)) {
  		f1_score_logitboost <- 0
	}
	print("LogitBoost Model Completed")

	#Linear SVM Model#
	print("Starting Linear SVM")
	#Linear SVM
	svm_linear_model <- train(x = data_df_train, 
	                   y = as.factor(response_train), 
	                   method = "svmLinear", 
	                   trControl = trctrl)
	svm_linear_predict <- predict(svm_linear_model, newdata = data_df_validation)
	svm_linear_confusion_matrix <- confusionMatrix(svm_linear_predict, 
	                                               as.factor(meta[-trainIndex, ]$class),
	                                               positive = '1', 
	                                               mode = "prec_recall")

	#get model statistics
	f1_score_linear <- svm_linear_confusion_matrix[[4]][[7]]
	precision_linear <- svm_linear_confusion_matrix[[4]][[5]]
	recall_linear <- svm_linear_confusion_matrix[[4]][[6]]
	accuracy_linear <- svm_linear_confusion_matrix[[3]][[1]]

	#replace F1 score with 0 if NA due to poor performance
	if (is.na(f1_score_linear)) {
	  f1_score_linear <- 0
	}

	print("Linear SVM Completed")
	end.time <- Sys.time() #record end time
	print(end.time - start.time) #check time taken to run on subsets of data

	#***Naive Model Performances***#
	model_list <- list(svm_linear_model,
	                   logitboost_model) #list of models to index through
	model_name <- c("Linear SVM",
	                "LogitBoost") #list of model types
	f1_score_internal_validation <- c(f1_score_linear, f1_score_logitboost) #list of f1 scores
	precision_internal_validation <- c(precision_linear, precision_logitboost) #list of precisions
	recall_internal_validation <- c(recall_linear, recall_logitboost) #list of recalls
	accuracy_internal_validation <- c(accuracy_linear, accuracy_logitboost) #list of unbalanced accuracies
	#create data frame with models as rows and measures as columns
	performances <- data.frame(model_name, 
							   f1_score_internal_validation,
							   precision_internal_validation,
							   recall_internal_validation,
							   accuracy_internal_validation)
	colnames(performances) <- c("Model", 
								"F1 Score (Training Set)", 
								"Precision (Training Set)", 
								"Recall (Training Set)",
								"Accuracy (Training Set)")
	print("Trained Model Performances")
	performances %>% arrange(desc(f1_score_internal_validation)) %>% print() #un-tuned models and their F1

	#save linear SVM model
	best_model <- svm_linear_model
	saveRDS(svm_linear_model, file = paste("SVM_Linear_Model_", num_tweets_min[i], ".rds", sep = ""))

	#save logitboost model
	saveRDS(logitboost_model, file = paste("LogitBoost_Model_", num_tweets_min[i], ".rds", sep = ""))

	#***********************************************************************************************************#
	#***TESTING***#
	print("Starting Testing")
	testing_data <- read.csv('~/Desktop/LHS_712/Final_Project/task2_en_validation.tsv', sep = '\t', header = TRUE, stringsAsFactors = FALSE)

	#remove stop words
	testing_data_tokenized_no_stops <- map_df(1:2, ~ unnest_tokens(testing_data, 
	                                                   word, 
	                                                   tweet, 
	                                                   token = "tweets")) %>%
	                                                   #n = .x)) %>%
	                             		anti_join(stop_words, by = "word") %>%
	                             		count(tweet_id, word, sort = TRUE)
	#replace brand names with the word "drug"
	#for (i in 1:nrow(testing_data_tokenized_no_stops)) {
	#if (testing_data_tokenized_no_stops[i, "word"] %in% brand_names$word) {
	#	testing_data_tokenized_no_stops[i, "word"] <- "drug" #replace brand name with drug
	#	}
	#}

	#count number of occurrences of each word in each tweet
	testing_data_counts <- testing_data_tokenized_no_stops 
							#%>% count(tweet_id, word, sort = TRUE)

	#right join, cast to document matrix, and use same features as training
	testing_data_dtm <- testing_data_counts %>%
	                    right_join(words_min, by = "word") %>% 
	                    bind_tf_idf(word, tweet_id, n) %>% 
	                    cast_dtm(tweet_id, word, tf_idf)

	#create intermediate data frame
	testing_meta <- tibble(tweet_id = as.numeric(dimnames(testing_data_dtm)[[1]])) %>% 
	                       left_join(testing_data[!duplicated(testing_data$tweet_id), ], 
	                                 by = "tweet_id")

	data_df_testing <- testing_data_dtm %>% as.matrix() %>% as.data.frame()
	data_df_testing <- na.omit(data_df_testing)
	rownames(data_df_testing) <- str_remove(rownames(data_df_testing), "X")

	#subset columns of testing data to those found in training data
	data_df_testing <- data_df_testing[, which(colnames(data_df_testing) %in% colnames(data_df_train))]

	#LogitBoost#
	#predict LogitBoost on testing set
	testing_predictions <- predict(logitboost_model, newdata = data_df_testing)
	levels(testing_predictions) <- c("0", "1") #change back to normal encoding
	predicted_labels <- data.frame(rownames(data_df_testing), testing_predictions) #create df with tweet ID and predicted label
	colnames(predicted_labels) <- c("tweet_id", "predicted_label")
	predicted_labels[, "tweet_id"] <- as.numeric(as.character(predicted_labels$tweet_id))

	#compare with actual labels for validation set
	actual_labels <- testing_data[c(which(testing_data$tweet_id %in% intersect(testing_data$tweet_id, str_remove(rownames(data_df_testing), "X"))), 3020), ] %>% select(tweet_id, class)
	colnames(actual_labels) <- c("tweet_id", "actual_label")
	predicted_actual_df <- left_join(predicted_labels, actual_labels)

	#generate confusion matrix
	tuned_logitboost_confusion_matrix_validation <- confusionMatrix(as.factor(predicted_actual_df$predicted_label), 
	                                                                as.factor(predicted_actual_df$actual_label),
	                                                                positive = '1', 
	                                                                mode = "prec_recall")
	#calculate F1 score
	f1_score_logitboost_oos <- tuned_logitboost_confusion_matrix_validation[[4]][[7]]
	print(paste("F1 Score of LogitBoost on Validation Set: ", f1_score_logitboost_oos, sep = ""))

	#calculate other statistics
	precision_logitboost_oos <- tuned_logitboost_confusion_matrix_validation[[4]][[5]]
	recall_logitboost_oos <- tuned_logitboost_confusion_matrix_validation[[4]][[6]]
	accuracy_logitboost_oos <- tuned_logitboost_confusion_matrix_validation[[3]][[1]] 

	#Linear SVM#
	#predict linear SVM on testing set
	testing_predictions <- predict(best_model, newdata = data_df_testing)
	levels(testing_predictions) <- c("0", "1") #change back to normal encoding
	predicted_labels <- data.frame(rownames(data_df_testing), testing_predictions) #create df with tweet ID and predicted label
	colnames(predicted_labels) <- c("tweet_id", "predicted_label")
	predicted_labels[, "tweet_id"] <- as.numeric(as.character(predicted_labels$tweet_id))

	#compare with actual labels for validation set
	actual_labels <- testing_data[c(which(testing_data$tweet_id %in% intersect(testing_data$tweet_id, str_remove(rownames(data_df_testing), "X"))), 3020), ] %>% select(tweet_id, class)
	colnames(actual_labels) <- c("tweet_id", "actual_label")
	predicted_actual_df <- left_join(predicted_labels, actual_labels)

	#generate confusion matrix
	tuned_svm_linear_confusion_matrix_validation <- confusionMatrix(as.factor(predicted_actual_df$predicted_label), 
	                                                                as.factor(predicted_actual_df$actual_label),
	                                                                positive = '1', 
	                                                                mode = "prec_recall")
	#calculate F1 score
	f1_score_linear_oos <- tuned_svm_linear_confusion_matrix_validation[[4]][[7]]
	print(paste("F1 Score of Linear SVM on Validation Set: ", f1_score_linear_oos, sep = ""))

	#calculate other statistics
	precision_linear_oos <- tuned_svm_linear_confusion_matrix_validation[[4]][[5]]
	recall_linear_oos <- tuned_svm_linear_confusion_matrix_validation[[4]][[6]]
	accuracy_linear_oos <- tuned_svm_linear_confusion_matrix_validation[[3]][[1]]

	#add to list of model statistics
	final_logitboost_f1_list[i] <- f1_score_logitboost_oos
	final_linear_f1_list[i] <- f1_score_linear_oos
	final_logitboost_precision_list[i] <- precision_logitboost_oos
	final_linear_precision_list[i] <- precision_linear_oos
	final_logitboost_recall_list[i] <- recall_logitboost_oos
	final_linear_recall_list[i] <- recall_linear_oos
	final_logitboost_accuracy_list[i] <- accuracy_logitboost_oos
	final_linear_accuracy_list[i] <- accuracy_linear_oos

	print("Testing Completed")
	print("****************")
	#***********************************************************************************************************#
}

##compare model performance OOS based on number of features and plot
num_tweets_vs_stats <- data.frame(num_tweets_min, 
							   unlist(num_features_list), 
							   unlist(final_logitboost_f1_list), 
							   unlist(final_linear_f1_list),
							   unlist(final_logitboost_precision_list),
							   unlist(final_linear_precision_list),
							   unlist(final_logitboost_recall_list),
							   unlist(final_linear_recall_list),
							   unlist(final_logitboost_accuracy_list),
							   unlist(final_linear_accuracy_list))
colnames(num_tweets_vs_stats) <- c("Num_Tweets_Cutoff_for_Feature", 
								"Num_Features", 
								"LogitBoost_F1_Score_Validation_Set", 
								"Linear_SVM_F1_Score_Validation_Set",
								"LogitBoost_Precision_Validation_Set",
								"Linear_SVM_Precision_Validation Set",
								"LogitBoost_Recall_Validation_Set",
								"Linear_SVM_Recall_Validation_Set",
								"LogitBoost_Accuracy_Validation_Set",
								"Linear_SVM_Accuracy_Validation_Set")
write.csv(num_tweets_vs_stats, file = "~/Desktop/LHS_712/Final_Project/Num_Tweets_vs_Stats.csv")
pdf("~/Desktop/LHS_712/Final_Project/Num_Features_vs_F1_Score_Linear_SVM.pdf")
ggplot(data = num_tweets_vs_stats) +
	geom_point(aes(x = Num_Features, y = Linear_SVM_F1_Score_Validation_Set)) +
	geom_line(aes(x = Num_Features, y = Linear_SVM_F1_Score_Validation_Set)) +
	xlab("Number of Features (Words)") +
	ylab("F1 Score on Validation Set")
dev.off()
pdf("~/Desktop/LHS_712/Final_Project/Num_Features_vs_F1_Score_LogitBoost.pdf")
ggplot(data = num_tweets_vs_stats) +
	geom_point(aes(x = Num_Features, y = LogitBoost_F1_Score_Validation_Set)) +
	geom_line(aes(x = Num_Features, y = LogitBoost_F1_Score_Validation_Set)) +
	xlab("Number of Features (Words)") +
	ylab("F1 Score on Validation Set")
dev.off()


