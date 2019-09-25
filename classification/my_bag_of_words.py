#!/usr/bin/env python
#coding:utf-8
## Author: Jiayi Chen 
## Time-stamp:  09/22/2018

import numpy as np
import scipy
import nltk, re, pprint
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from nltk import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from replacers import RegexpReplacer
from replacers import RepeatReplacer
#import json
import string

def list_to_dict(list):
    keyword = list
    indices=range(len(list))
    return dict(zip(keyword,indices))

def read_from_file(file_name):
    f=open(file_name, 'rU')
    corpus=[]
    for line in f: # read a line from f
        corpus.append(line.strip())
    return corpus

def ExtWd_preprocessing(sentence):
    """
    Separate words for a sample (or, a sentence), while pre-processing
    """
    tokens=word_tokenize(sentence)

    stopset = set(stopwords.words('english'))
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    s = nltk.stem.SnowballStemmer('english')
    stemmer=PorterStemmer()
    replacerReplacer=RepeatReplacer()
    words_cleaned=[]

    for w in tokens:
        if w.isalpha()==True: # 2) check if all the characters in a word is alphabetic
            w=replacerReplacer.replace(w)# 5) replace repeating characters
            w=w.lower() # 1) lowercase
            if w not in stopset:  # 3) remove stopwords
                words_cleaned.append(stemmer.stem(lemmatizer.lemmatize(w)))# 4) stem words
    return words_cleaned





def build_vocabulary(sentences,max_df,min_df):

    words = []
    i=0
    for sentence in sentences:
        w = ExtWd_preprocessing(sentence)
        words.extend(w)
        i+=1

        print "scan the ",i,"th sample"
    print 'After collecting all words: vocabulary volume is ', len(words)
    fdist = FreqDist(words)
    # fdist.plot()
    vocabulary0=fdist.keys()

    # # remove
    # Most_freq_list=[]
    # for word, frequency in fdist.most_common(int(max_df*len(vocabulary0))):
    #     Most_freq_list.append(word)
    #
    # No_LeastFreqWords_list = []
    # for word, frequency in fdist.most_common(int((1-min_df) * len(vocabulary0))):
    #     No_LeastFreqWords_list.append(word)
    #
    # vocabulary=[]
    # for w in No_LeastFreqWords_list:
    #     if w not in Most_freq_list:
    #         vocabulary.append(w)

    vocabulary = []
    for w in vocabulary0: # 5) preprocessing trick: discard low-frequency words
        if fdist[w] > 3:
            vocabulary.append(w)

    print 'After re-scanning all words: vocabulary volume is ', len(vocabulary)


    fileObject = open('vocabulary_saved.txt', 'w')
    for ip in vocabulary:
        fileObject.write(ip)
        fileObject.write('\n')
    fileObject.close()


    return vocabulary,len(vocabulary)

# build vocabulary from training data
def buid_or_read_voc(trn_x):
    """
    build vocabulary -  vocabulary,n_vocabulary = build_vocabulary(trn_x)# including "?,!#$%^&*("
    read existed vocabulary - from file
    """
    # vocabulary, n_vocabulary = build_vocabulary(trn_x,0.01,0.01)
    vocabulary=[]
    voc_file = open("vocabulary_saved.txt", "r")
    for line in voc_file:
        line = line.rstrip("\n")
        vocabulary.append(line)
    n_vocabulary=len(vocabulary)
    voc_file.close()

    return vocabulary,n_vocabulary




def bag_of_words_all(sentences, n_sample,vocabulary_dict):
    bag= np.zeros([n_sample,len(vocabulary_dict)],float)
    for n in range(n_sample):
        if (n+1)%200==0:
            print '         (I have built',n*100.0/n_sample,'% BOW Matrix', n,')'
        sentence_tokens = ExtWd_preprocessing(sentences[n])
        i = 0
        for word in sentence_tokens:
            try:
                index=vocabulary_dict[word]
                bag[n, index] += 1.0
                i += 1
            except:
                pass
    return np.array(bag)


def setup_feature_vector_from_bag(vocab, bag, y):
    Fxy = np.zeros(2 * len(bag) + 1, float)  # faster to converge
    if y == '0':
        Fxy[:len(bag)]=bag
    elif y=='1':
        Fxy[len(bag): (2 * len(bag))]=bag
    Fxy[2 * len(bag)] = 1
    # for j in range(len(bag)):
    #     if y == '0':
    #         Fxy[2 * j] = bag[j]
    #     elif y == '1':
    #         Fxy[2 * j + 1] = bag[j]
    # Fxy[2 * len(bag)] = 1
    return np.array(Fxy)

