library(tidyverse)
library(tm)
library(cld2)
library(tidytext)
library(topicmodels)

#Load reviews data
review.raw <- read_csv("../data/reviews.csv")


#Filter out the no English reviews
review.en <- review.raw %>%
  mutate(cld2 = detect_language(comments, plain_text = FALSE)) %>%
  filter(cld2 == "en") 

review.df <- aggregate(review.en$comments, list(review.en$listing_id), paste, collapse=" ")
names(review.df) <- c("id", "comments")

review.raw.corp <- VCorpus(VectorSource(review.df$comments))

#Basic clean corpus function (without stop word remove)
clean_corpus <- function(corpus) {
  # Remove punctuation
  corpus <- tm_map(corpus, removePunctuation)
  # Transform to lower case
  corpus <- tm_map(corpus, content_transformer(tolower))
  # Strip whitespace
  corpus <- tm_map(corpus, stripWhitespace)
  # Remove Numbers
  corpus <- tm_map(corpus, removeNumbers)
  return(corpus)
}

review.crop <- clean_corpus(review.raw.corp)

review.df.clean <-data.frame(id = review.df$id, text = sapply(review.crop,as.character),stringsAsFactors = FALSE)

review.tib <- review.df.clean %>%
  unnest_tokens(word, text) %>%
  count(id, word, sort=TRUE)

review.tib.idf <- review.tib %>%
  bind_tf_idf(word, id, n) %>%
  arrange(desc(tf_idf))

# Create a TermDocument Matrix
review.dtm <- review.tib.idf %>%
  cast_dtm(id, word, n)

review.df.clean <- arrange(review.df.clean, id)
review.lda <- LDA(review.dtm, k = 4, control = list(seed = 1948))

review.topics <- tidy(review.lda, matrix = "gamma")
review.topics$document <- as.numeric(review.topics$document)
review.topics <- arrange(review.topics, document, topic)

review.feature <- as.data.frame(matrix(review.topics$gamma, ncol = 4))
review.feature <- cbind(review.df.clean$id,review.feature)
names(review.feature) <- c("id", "ReviewF1", "ReviewF2", "ReviewF3", "ReviewF4")

save(review.feature, file = "../output/ReviewFeature.RData")
save(review.df.clean, file = "../output/CleanReviewDf.RData")
save(review.tib, file = "../output/CleanReviewTib.RData")
save(review.dtm, file = "../output/CleanReviewDTM.RData")