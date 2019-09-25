#!/usr/bin/env python
#coding:utf-8

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression as LR
from time import time

# Load data
trn_texts = open("trn.data").read().strip().split("\n")
trn_labels = map(str, open("trn.label").read().strip().split("\n"))
print "Training data ..."
print len(trn_texts), len(trn_labels)

dev_texts = open("dev.data").read().strip().split("\n")
dev_labels = map(str, open("dev.label").read().strip().split("\n"))
print "Development data ..."
print len(dev_texts), len(dev_labels)

tst_texts = open("tst.data").read().strip().split("\n")
print "Test data ..."
print len(tst_texts)



vocab_filter_choice = 'ngram_range12'
# clf_para='l2_tune_C'
C_parameters=[ 400,0.25]
clf_para='l1_tune_C'
C_l2_best = 0.17


# define a vectorization option
if vocab_filter_choice == 1:
    print "Preprocessing without any feature selection"
    vectorizer = CountVectorizer(lowercase=False)
    # vocab size 60641
elif vocab_filter_choice == 'default':
    print "Lowercasing all the tokens"
    vectorizer = CountVectorizer()
    # vocab size 47963
elif vocab_filter_choice == 2.5:# default
    vectorizer = CountVectorizer(lowercase=True)
    # vocab size 47963
elif vocab_filter_choice == 'ngram_range12':
    print "Uni- and bi-gram"
    vectorizer = CountVectorizer(ngram_range=(1, 2))
elif vocab_filter_choice == 3:
    print "Lowercasing and filtering out low-frequency words"
    vectorizer = CountVectorizer(lowercase=True, min_df=2)
    # vocab size 22492
elif vocab_filter_choice == 4:
    print "Lowercasing and filtering out low-frequency words, uni- and bi-gram"
    vectorizer = CountVectorizer(lowercase=True, min_df=2, ngram_range=(1,2))
    # vocab size 239158, rich feature set

    # vocab 1048596
elif vocab_filter_choice == 6:
    print "Lowercasing and filtering out high-frequency words"
    vectorizer = CountVectorizer(lowercase=True, max_df=0.5)
    # vocab size 60610



# vectoriz data     # 矩阵 [n_samples, vocab_size] 统计词频
trn_data = vectorizer.fit_transform(trn_texts)
print trn_data.shape # (30000, 47963)
dev_data = vectorizer.transform(dev_texts)
print dev_data.shape # (10000, 47963)
tst_data = vectorizer.transform(tst_texts)
print tst_data.shape # (10000, 47963)


# define a LR classifier ( more regularization options )
if clf_para=='default' :
    classifier = LR()
elif clf_para=='l1_tune_C' :
    for C in C_parameters:
        classifier = LR(C=C,penalty='l1')  # default
        print "When lamda = ", 1.0/C
        t1=time()
        classifier.fit(trn_data, trn_labels)
        print "    Training time is ", time()-t1,' seconds'
        print "    Training accuracy =", classifier.score(trn_data, trn_labels)
        print "    Dev accuracy =", classifier.score(dev_data, dev_labels)
elif clf_para == 'l2_tune_C':
    for C in C_parameters:
        classifier = LR(C=C)  # default
        print "When lamda = ", 1.0/C
        t1=time()
        classifier.fit(trn_data, trn_labels)
        print "    Training time is ", time()-t1,' seconds'
        print "    Training accuracy =", classifier.score(trn_data, trn_labels)
        print "    Dev accuracy =", classifier.score(dev_data, dev_labels)


# #
# classifier.fit(trn_data, trn_labels)
#
#
# # Measure the performance on training and dev data
# print "Training accuracy =", classifier.score(trn_data, trn_labels)
# print "Dev accuracy =", classifier.score(dev_data, dev_labels)
#
#
# # different combinations of hyperparameters