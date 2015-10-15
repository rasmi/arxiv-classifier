# arXiv-classification.py
# By Rasmi Elasmar
# Modeling Social Data
# 05/08/2015

import pandas as pd
import numpy as np
import requests
import json
import os
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

###############################################################################

####################
# DATA PREPARATION #
####################

###############################################################################
categories = ['astro-ph.CO',
              'astro-ph.EP',
              'astro-ph.GA',
              'astro-ph.HE',
              'astro-ph.IM',
              'astro-ph.SR']

# Download articles given astro-ph category.
def download_articles(category):
    url = 'http://export.arxiv.org/api/query?'
    params = {
        'search_query': 'cat:' + category,
        'start': 0,
        'max_results': 2000
    }
    queryparams = '&'.join([param+'='+str(value) for param, value in params.items()])
    query = url + queryparams
    results = requests.get(query)
    soup = BeautifulSoup(results.text)
    entries = soup.findAll('entry')
    
    def title(entry):
        return entry.findAll('title')[0].text
    
    def summary(entry):
        return entry.findAll('summary')[0].text
    
    def authors(entry):
        # Author item for every author with name sub-items.
        return ', '.join([author.findAll('name')[0].text for author in entry.findAll('author')])
    
    def article_category(entry):
        # <category scheme="http://arxiv.org/schemas/atom" term="astro-ph.CO"></category>
        return [cat['term'] for cat in entry.findAll('category') if 'astro-ph' in cat['term']][0]
    
    def pubdate(entry):
        return entry.findAll('published')[0].text
    
    def idurl(entry):
        return entry.findAll('id')[0].text
    
    def get_article(entry):
        return {'title': title(entry),
                'summary': summary(entry),
                'authors': authors(entry),
                'category': article_category(entry),
                'pubdate': pubdate(entry),
                'idurl': idurl(entry)}
    
    articles = []
    
    for entry in entries:
        try:
            articles.append(get_article(entry))
        except:
            print "Parsing article failed in category " + category
    
    jsondata = {'category': category, 'articles': articles}
    with open('articles/'+category+'.json', 'w') as jsonfile:
        json.dump(jsondata, jsonfile)

# Load all articles into memory.
# Use local files if available in articles/ folder,
# otherwise download from arXiv.org.
all_articles = {}
for category in categories:
    if category+'.json' not in os.listdir('articles'):
        print "Downloading " + category
        download_articles(category)
    
    print "Loading " + category 
    with open('articles/'+category+'.json', 'r') as jsonfile:
        categoryarticles = json.load(jsonfile)['articles']
        
    all_articles[category] = categoryarticles

# Truncate to size of category with minimum number of articles available.
size = min([len(articles) for articles in all_articles.values()])
for category, values in all_articles.items():
    all_articles[category] = values[:size]

# Combine all article summaries and categories into a DataFrame.
summaries = pd.DataFrame(columns=['summary','category'])

def get_summaries(articles):
    summaries = [article['summary'] for article in articles]
    categories = [article['category'] for article in articles]
    return pd.DataFrame({'summary': summaries, 'category': categories})

for articles in all_articles.values():
    summaries = summaries.append(get_summaries(articles))

# Shuffle so categories aren't grouped.
summaries = summaries.reset_index(drop=True)
summaries = summaries.reindex(np.random.permutation(summaries.index))

###############################################################################

######################
# CLASSIFY SUMMARIES #
######################

###############################################################################

# Split shuffled DF into 50/50 train/test data.
summaries_train = summaries[:len(summaries)/2]
summaries_test = summaries[len(summaries)/2:]


###############
# NAIVE BAYES #
###############

# Use CountVectorizer.fit_transform() to get vocabulary and transform into sparse matrix.
count_vectorizer = CountVectorizer()
sparse_counts_train = count_vectorizer.fit_transform(summaries_train['summary'].values)

sparse_counts_test = count_vectorizer.transform(summaries_test['summary'].values)

# Classify data using MultinomialNB
classifier = MultinomialNB()
categories_train = summaries_train['category'].values
classifier.fit(sparse_counts_train, categories_train)

predictions = classifier.predict(sparse_counts_test)
confusion = confusion_matrix(summaries_test['category'].values, predictions)
print "NAIVE BAYES RESULTS"
print "Accuracy: %f"%(np.sum(predictions == summaries_test['category'].values)/(len(predictions)*1.0))
print "Confusion:\n", confusion

# Plot large confusion matrix to save for report.
confusionplot, ax = plt.subplots(figsize=(10,10))
ax.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
category_labels = list(np.sort(summaries_test['category'].unique()))
category_labels = [label.replace('astro-ph.', '') for label in category_labels]
ticks = np.arange(len(category_labels))
plt.xticks(ticks, category_labels, fontsize=14)
plt.yticks(ticks, category_labels, fontsize=14)
plt.ylabel('True', fontsize=16)
plt.xlabel('Predicted', fontsize=16)
plt.title("Confusion", fontsize=18)
plt.savefig('confusion.png')


###########################
# NAIVE BAYES WITH TF-IDF #
###########################

# Use CountVectorizer.fit_transform() to get vocabulary and transform into sparse matrix.
# Also apply tf-idf for any possible performance improvements.
count_vectorizer_tfidf = CountVectorizer()
sparse_counts_train_tfidf = count_vectorizer_tfidf.fit_transform(summaries_train['summary'].values)
tfidf_transformer = TfidfTransformer()
sparse_counts_train_tfidf = tfidf_transformer.fit_transform(sparse_counts_train_tfidf)

sparse_counts_test_tfidf = count_vectorizer_tfidf.transform(summaries_test['summary'].values)
sparse_counts_test_tfidf = tfidf_transformer.transform(sparse_counts_test_tfidf)

# Classify data using MultinomialNB
classifier_tfidf = MultinomialNB()
categories_train_tfidf = summaries_train['category'].values
classifier_tfidf.fit(sparse_counts_train_tfidf, categories_train_tfidf)

predictions_tfidf = classifier.predict(sparse_counts_test_tfidf)
confusion_tfidf = confusion_matrix(summaries_test['category'].values, predictions_tfidf)
print "NAIVE BAYES WITH TF-IDF RESULTS"
print "Accuracy: %f"%(np.sum(predictions_tfidf == summaries_test['category'].values)/(len(predictions_tfidf)*1.0))
print "Confusion:\n", confusion_tfidf
plt.imshow(confusion_tfidf, interpolation='nearest', cmap=plt.cm.Blues)


############################
# NAIVE BAYES WITH N-GRAMS #
############################

# Use CountVectorizer.fit_transform() to get vocabulary and transform into sparse matrix.
# Use n-grams to see if phrases help.
count_vectorizer_ngram = CountVectorizer(ngram_range=(1, 3))
sparse_counts_train_ngram = count_vectorizer_ngram.fit_transform(summaries_train['summary'].values)

sparse_counts_test_ngram = count_vectorizer_ngram.transform(summaries_test['summary'].values)

# Classify data using MultinomialNB
classifier_ngram = MultinomialNB()
categories_train_ngram = summaries_train['category'].values
classifier_ngram.fit(sparse_counts_train_ngram, categories_train_ngram)

predictions_ngram = classifier_ngram.predict(sparse_counts_test_ngram)
confusion_ngram = confusion_matrix(summaries_test['category'].values, predictions_ngram)
print "NAIVE BAYES WITH N-GRAMS RESULTS"
print "Accuracy: %f"%(np.sum(predictions_ngram == summaries_test['category'].values)/(len(predictions_ngram)*1.0))
print "Confusion:\n", confusion_ngram
plt.imshow(confusion_ngram, interpolation='nearest', cmap=plt.cm.Blues)
