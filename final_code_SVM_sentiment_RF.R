library("tm")
library("RTextTools")
library(dplyr) #data manipulation
library(tidytext) # use stop words  #text mining
library(tokenizers)
library(tidyr)
library(tictoc)
library(tibble)
library(devtools)
library(ggplot2) #visualizations
library(gridExtra) #viewing multiple plots together
library(wordcloud2) #creative visualizations
library(stringr)
library(pander)
library(modelr) # For splitting into training, validation, and testing sets
library(rsample) # split the data
library(ISLR) # might delete later
library(caret)
library(stringr)
library(e1071)
library(MLmetrics)
library(quanteda)
library(LiblineaR)
set.seed(123)
#############
#define some colors to use throughout the analysis
my_colors <- c("#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#D55E00")


input <- read.table("code/data/training.csv", sep=",", header = TRUE, stringsAsFactors = F)
training_all <- input %>% 
  mutate(tweet = gsub('http\\S+\\s*', '', tweet)) %>% 
  ## Remove Hashtags - the special character only
  mutate(tweet = gsub('#', '', tweet)) %>% 
  ## Convert Mentions into a code
  mutate(tweet = gsub('@\\S+', '__username__', tweet)) %>% 
  ## Convert retweet Mentions into a code
  mutate(tweet = gsub('rt __username__', '__retweet__', tweet))

# Basic cleaning
# function to expand contractions in an English-language source
fix.contractions <- function(doc) {
  # "won't" is a special case as it does not expand to "wo not"
  doc <- gsub("won't", "will not", doc)
  doc <- gsub("n't", " not", doc)
  doc <- gsub("'ll", " will", doc)
  doc <- gsub("'re", " are", doc)
  doc <- gsub("'ve", " have", doc)
  doc <- gsub("'m", " am", doc)
  doc <- gsub("'d", " would", doc)
  # 's could be 'is' or could be possessive: it has no expansion
  doc <- gsub("'s", "", doc)
  return(doc)
}

# fix (expand) contractions
training_all$tweet <- sapply(training_all$tweet, fix.contractions)

# To be consistent, go ahead and convert everything to lowercase with the handy tolower() function.
# convert everything to lower case
training_all$tweet <- sapply(training_all$tweet, tolower)

# Review standard stop words by calling stopwords("en").
stopwords("en")

###! Using the c() function allows you to add new words to the stop words list. 
# For example, the following would add "word1" and "word2" to the default list of English stop words:
# all_stops <- c("word1", "word2", stopwords("en"))
# all_stops <- stopwords("en")

training_all$tweet
#unnest and remove stop, undesirable and short words
# In the stop_words data object in tidytext, 
# the column is called word and in your dataframe, it is called token
## In this step, we lost the emojis and :)'s
training_all_filtered <- training_all %>%
  unnest_tokens(word, tweet) %>%
  anti_join(stop_words) %>% 
  distinct() %>% 
  filter(word != "__username__")

#filter(nchar(word) > 3) # to filter the characters < 3

training_all_filtered %>% 
  filter(class == 1) %>% 
  View()

training_all_filtered %>% 
  filter(class == 0) %>% 
  View()

# After filtering the "username", the word cloud we got
training_all_counts<- training_all_filtered %>%
  count(word, sort = TRUE) %>% 
  filter(word != "__username__")

## Sentiment analysis can be done as an inner join. 
# Sentiment lexicons are available via the get_sentiments() function. 
# Let’s look at the words with a positive score from the lexicon of Bing Liu and collaborators. 
# What are the most common positive words in tweet 1?

positive <- get_sentiments(lexicon = "bing") %>%
  filter(sentiment == "positive")

training_all_filtered %>%
  filter(Label == 1) %>%
  semi_join(positive) %>%
  count(word, sort = TRUE)

# examine how sentiment changes during each tweet
## Sentiment analysis can be done as an inner join. 
# Sentiment lexicons are available via the get_sentiments() function. 
# Let’s look at the words with a positive score from the lexicon of Bing Liu and collaborators. 
# What are the most common positive words in tweet 1?

bing <- get_sentiments(lexicon =c("bing"))

training_sentiments <- training_all_filtered %>%
  inner_join(bing) %>%
  count(Label, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative)

training_all_with_sentiment <-  merge(x = training_sentiments, y = training_all, by = "Label", all.y=TRUE)   

# Plot sentiment scores
ggplot(training_all_with_sentiment, aes(Label, sentiment, fill = class)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  facet_wrap(~class, ncol = 2, scales = "free_x")

bing_word_counts <- training_all_filtered %>%
  inner_join(bing) %>%
  count(word, sentiment, sort = TRUE)

# This can be shown visually, and we can pipe straight into ggplot2 
# because of the way we are consistently using tools built for handling tidy data frames.

bing_word_counts %>%
  filter(n > 50) %>%
  mutate(n = ifelse(sentiment == "negative", -n, n)) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n, fill = sentiment)) +
  geom_col() +
  coord_flip() +
  labs(y = "Contribution to sentiment")

# Add the sentiment analysis result to the original dataset

# Merge with selected columns
# You can do this by subsetting the data you pass into your merge:
#t <-  merge(x = training_sentiments, y = training_all[ , c("class", "tweet", "Label")], by = "Label", all.x=TRUE)

training_all_with_sentiment <-  merge(x = training_sentiments, y = training_all, by = "Label", all.y=TRUE)   


### Creaet and Inspect a Document Term Matrix
# tm has functions that help prepare text data for modeling. 
# One essential data structure is the Document Term Matrix. 
# In a DTM, every document is a row and every column is a word. 
# Like tidytext, tm has a number of built-in functions to find 
# frequent words, inspect the matrix, etc. For instance, inspect() 
# reports the number of rows in our corpus , 
# the number of words  and the length of the longest document . 
# 
# For our model, we exclude terms (words) with fewer than 2 appearances 
# in the messages.

# Output of the tm inspect() function summarizes the number of rows 
# and terms in the matrix and displays the first 10 rows and columns.

## cleanup steps ##
View(training_all_with_sentiment)

# clean text and create DFM
twcorpus <- corpus(training_all_with_sentiment$tweet)
View(twcorpus)
twdfm <- dfm(twcorpus, remove=stopwords("english"), remove_url=TRUE,
             verbose=TRUE)

twdfm <- dfm_trim(twdfm, min_docfreq = 2, verbose=TRUE)



### Split and save the Validation set
# training and test sets
set.seed(123)
training <- sample(1:nrow(training_all_with_sentiment), floor(.80 * nrow(training_all_with_sentiment)))
valid <- (1:nrow(training_all_with_sentiment))[1:nrow(training_all_with_sentiment) %in% training == FALSE]

library(SparseM)
fit <- svm(x=twdfm[training,], y=factor(training_all_with_sentiment$class[training]),
           kernel="linear", cost=10, probability=TRUE)

# predict the validation set
preds <- predict(fit, twdfm[valid,])

## function to compute accuracy
accuracy <- function(ypred, y){
  tab <- table(ypred, y)
  return(sum(diag(tab))/sum(tab))
}
# function to compute precision
precision <- function(ypred, y){
  tab <- table(ypred, y)
  return((tab[2,2])/(tab[2,1]+tab[2,2]))
}
# function to compute recall
recall <- function(ypred, y){
  tab <- table(ypred, y)
  return(tab[2,2]/(tab[1,2]+tab[2,2]))
}

# confusion matrix
table(preds, training_all_with_sentiment$class[valid])

# performance metrics
accuracy(preds, training_all_with_sentiment$class[valid])

precision(preds, training_all_with_sentiment$class[valid])

recall(preds, training_all_with_sentiment$class[valid])

F1_Score(preds, training_all_with_sentiment$class[valid], positive = "1")

# Tuning hyperparameters
fitControl <- trainControl(method = "repeatedcv",
                           number = 3,
                           repeats = 3,
                           search = "grid")

fit <- svm(x=twdfm[training,], y=factor(training_all_with_sentiment$class[training]),
           kernel="linear", cost=10, probability=TRUE)

svm_mod <- train(x = twdfm[training,],
                 y = factor(training_all_with_sentiment$class[training]),
                 method = "svmLinearWeights2",
                 trControl = fitControl,
                 tuneGrid = data.frame(cost = 0.01, 
                                       Loss = 0, 
                                       weight = seq(0.5, 1.5, 0.1)))
plot(svm_mod)
# data_dtm <- training_all_with_sentiment %>%
#   cast_dtm(Label, tweet, sentiment)


### Random Forest
### Split and save the Validation set
auto_split <- initial_split(data = training_all_with_sentiment, prop = 0.8)
auto_train <- training(auto_split)
auto_test <- testing(auto_split)

# split into training and validation sets
second_split <- initial_split(data = auto_train, prop = 0.8)
train_training <- training(second_split)
train_validation <- testing(second_split)

tweet_tokens <- train_training  %>%
  unnest_tokens(output = word, input = tweet) %>%
  # remove numbers
  filter(!str_detect(word, "^[0-9]*$")) %>%
  # # remove stop words
  # anti_join(stop_words) %>%
  # stem the words
  # use the Porter stemming algorithm
  mutate(word = SnowballC::wordStem(word)) %>% 
  filter(nchar(word) > 3)

### Create document-term matrix
# Tidy text data frames are one-row-per-token, 
# but for statistical learning algorithms we need our data in a 
# one-row-per-document format. That is, a document-term matrix. 
# We can use cast_dtm() to create a document-term matrix.

tweet_dtm <- tweet_tokens %>%
  # get count of each token in each document
  count(Label, word) %>%
  # create a document-term matrix with all features and tf weighting
  cast_dtm(document = Label, term = word, value = n)


### Sparsity
# removeSparseTerms(tweet_dtm, sparse = .999)

tweet_dtm <- removeSparseTerms(tweet_dtm, sparse = .999)

### Estimate model############
tweet_rf <- train(x = as.matrix(tweet_dtm),
                  y = factor(train_training$class),
                  method = "ranger",
                  num.trees = 200,
                  trControl = trainControl(method = "oob"))

#### prepare the validation set###
tweet_tokens_valid <- train_validation  %>%
  unnest_tokens(output = word, input = tweet) %>%
  filter(!str_detect(word, "^[0-9]*$")) %>%
  mutate(word = SnowballC::wordStem(word)) %>% 
  filter(nchar(word) > 3)

tweet_dtm_valid <- tweet_tokens_valid %>%
  count(Label, word) %>%
  cast_dtm(document = Label, term = word, value = n)

tweet_dtm_valid <- removeSparseTerms(tweet_dtm_valid, sparse = .999)

preds <- predict(tweet_rf, data = as.matrix(tweet_dtm_valid))

print(tweet_rf)
plot(tweet_rf)

# Use "ranger" to implement the random forest model
# Much faster and more efficient than the standard rf model
# Due to the number of variables (tokens) in the model

#  Below: it shows a random forest model with 10 trees, compared to a more
# typical random forest model with 200 trees

# some documents are lost due to not having any relevant tokens after tokenization
# make sure to remove their associated labels so we have the same number of observations
tweet_slice <- slice(train_training, as.numeric(tweet_dtm$dimnames$Docs))
tic()
tweet_rf_10 <- train(x = as.matrix(tweet_dtm),
                     y = factor(train_training$class),
                     method = "ranger",
                     num.trees = 10,
                     importance = "impurity",
                     trControl = trainControl(method = "oob"))
toc()

tweet_rf_10$finalModel

