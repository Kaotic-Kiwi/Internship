# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:09:51 2019

@author: straw
"""

# This is a basic module why do I have to load it manually ...?
from collections import defaultdict

import numpy as np
import pandas as pd
from pprint import pprint
import dfply as dpy
import nltk
import gensim

from sklearn.model_selection import train_test_split

import timeit

import matplotlib.pyplot as plt

# Import data
reviews = pd.read_csv('C:/Users/straw/Desktop/stageM2/scripts-francesca/reviews.csv') >> dpy.select("Restaurant_ID", "Review_ID", "Review_TEXT")

# Only kept the restaurant 'FR0210153861525'
reviews =  reviews[reviews['Restaurant_ID'] == 'FR0210153861525'].reset_index(level=0,drop=True) >> dpy.drop("Restaurant_ID")

reviews.index = reviews['Review_ID']
reviews = reviews >> dpy.drop("Review_ID")

# Lower case
reviews['Review_TEXT'] = reviews['Review_TEXT'].apply(lambda x: " ".join(x.lower() for x in x.split()))
reviews['Review_TEXT'].head()

# Tokenization function
def tokenizes(text):
    tokenizer = nltk.RegexpTokenizer(r'[a-zà-ÿœ0-9][a-zà-ÿœ0-9]+')
    return (tokenizer.tokenize(text))

# Tokenize reviews into a data.frame
new_df = (reviews['Review_TEXT'].apply(tokenizes)
                                .apply(pd.Series)
                                .stack()
                                .reset_index(level=0)
                                .set_index('Review_ID')
                                .rename(columns={0: 'word'}))

new_df['doc'] = new_df.index
new_df = new_df.reset_index(drop=True)


# Deleting anyword that contains "apostrophe"
new_df['word'] = new_df['word'].str.replace('[^\w\s]','')
new_df = new_df[~new_df['word'].str.contains('\s')]
#new_df = new_df[~new_df['word'].str.contains('[^\w\s]')]


# This file contains way more stop than what I'm collecting above
sw = pd.read_table('C:/Users/straw/Desktop/stageM2/scripts-francesca/fr_stopwords.txt', header=None)
sw

# Deleting stop words and numbers
new_df = new_df.loc[~new_df['word'].isin(sw[0])]
new_df = new_df[~new_df['word'].str.contains('\d')]
new_df.reset_index(inplace=True, drop=True)

###################################
## Filtering words that occure more than once and less than 300.

# Total frequence of each word
total_freq = new_df.groupby(['word']).size().to_frame('count').reset_index() >> dpy.arrange('count', ascending=False)
total_freq.reset_index(inplace=True, drop=True)


# The words to keep depending on their frequence
ok_words = total_freq['word'][(total_freq['count'] < 300) & (total_freq['count'] > 1)]
new_df_flt = new_df.loc[new_df['word'].isin(ok_words)]

##########################################
## Back to LDA and perplexity

# List of lists of the data.set
list_of_docs_flt = new_df_flt.groupby('doc')['word'].apply(list)

## Splitting data into train and test
list_of_docs_flt_train, list_of_docs_flt_test = train_test_split(list_of_docs_flt, test_size=0.2, random_state=0)

# Creating dictionary for each set
dictionary_train_flt = gensim.corpora.Dictionary(list_of_docs_flt_train)
#dictionary_test_flt = gensim.corpora.Dictionary(list_of_docs_flt_test)

# DTM matrix for each set
doc_term_matrix_train_flt = [dictionary_train_flt.doc2bow(text) for text in list_of_docs_flt_train]
doc_term_matrix_test_flt = [dictionary_train_flt.doc2bow(text) for text in list_of_docs_flt_test]

print(doc_term_matrix_train_flt[0])
print(doc_term_matrix_test_flt[0])

# LDA on train set
ldamodel_train_flt = gensim.models.ldamodel.LdaModel(doc_term_matrix_train_flt, num_topics=4, id2word = dictionary_train_flt, passes=20, alpha='auto', per_word_topics=True)

# If you see the same keywords being repeated in multiple topics, it’s probably a sign that the ‘k’ is too large.
pprint(ldamodel_train_flt.print_topics(num_topics=4, num_words=3))

# "Fake" perplexity
perplex = ldamodel_train_flt.bound(doc_term_matrix_test_flt)
print("Perplexity: %s" % perplex)

# Compute Perplexity
print('\nPerplexity: ', ldamodel_train_flt.log_perplexity(doc_term_matrix_test_flt))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = gensim.models.coherencemodel.CoherenceModel(model=ldamodel_train_flt, texts=list_of_docs_flt_test, dictionary=dictionary_train_flt, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


########################################

grid_flt = defaultdict(list)

 # num topics
parameter_list=[2, 5, 10, 15, 20, 25, 30]


for parameter_value in parameter_list:
    print("starting pass for parameter_value = %.3f" % parameter_value)
    start_time = timeit.default_timer()
    # run model
    ldamodel_train_flt = gensim.models.ldamodel.LdaModel(corpus=doc_term_matrix_train_flt, id2word = dictionary_train_flt, num_topics = parameter_value, passes=25, per_word_topics=True)

    # show elapsed time for model
    elapsed = timeit.default_timer() - start_time
    print("Elapsed time: %s" % elapsed)
    
    # Compute perplexity
    perplex =  ldamodel_train_flt.log_perplexity(doc_term_matrix_test_flt)
    print("Perplexity score: %s" % perplex)
    grid_flt[parameter_value].append(perplex)
    
    # Compute Coherence Score
    coherence_model_lda = gensim.models.coherencemodel.CoherenceModel(model=ldamodel_train_flt, texts=list_of_docs_flt_test, dictionary=dictionary_train_flt, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print("Coherence Score: %s" % coherence_lda)
    grid_flt[parameter_value].append(coherence_lda)
    
    
# Let's plot everything
saved_flt = pd.DataFrame([(k, v[0], v[1]) for k, v in grid_flt.items()], columns=['k', 'Perplexity', 'Coherence'])

plt.figure(0)
plt.subplot(1,2,1)
plt.plot(saved_flt['k'], saved_flt['Perplexity'])
plt.title('Perplexity score')
plt.subplot(1,2,2)
plt.plot(saved_flt['k'], saved_flt['Coherence'])
plt.title('Coherence score')
plt.suptitle("Criteria variation for the restaurant FR0210153861525 (filtered words)")




