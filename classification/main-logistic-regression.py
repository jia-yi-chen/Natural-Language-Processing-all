#!/usr/bin/env python
#coding:utf-8
## Author: Jiayi Chen 
## Time-stamp:  09/22/2018

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



# vocab_filter_choice = 'ngram_range12'
vocab_filter_choice = 'tune_best_model'
# clf_para='l2_tune_C'
clf_para='tune_best_model'
C_parameters=[ 10,5,2,1,0.8,0.5,0.3,0.2,0.15,0.1]
# clf_para='l1_tune_C'
# C_l2_best = 0.17
C_best=1.0/7.15


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

elif vocab_filter_choice == 'tune_best_model':
    # vectorizer = CountVectorizer(lowercase=True,min_df=3,ngram_range=(1,2),stop_words='english')
    vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 2),min_df=3, max_df=0.5)



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

elif clf_para == 'tune_best_model':
    classifier = LR(C=C_best,penalty='l2',solver='lbfgs')  # default
    # for C in C_parameters:
    #     classifier = LR(C=C,penalty='l2',solver='lbfgs',max_iter=100)  # default
    #     print "When lamda = ", 1.0 / C
    #     t1 = time()
    #     classifier.fit(trn_data, trn_labels)
    #     print "    Training time is ", time() - t1, ' seconds'
    #     print "    Training accuracy =", classifier.score(trn_data, trn_labels)
    #     print "    Dev accuracy =", classifier.score(dev_data, dev_labels)
    #     # prediction
    #     tst_label = classifier.predict(tst_data)
    #     print "Test results =", tst_label
    #     fileObject = open('lr-test.pred', 'w')
    #     for label in tst_label:
    #         fileObject.write(label)
    #         fileObject.write('\n')
    #     fileObject.close()

#
classifier.fit(trn_data, trn_labels)


# Measure the performance on training and dev data
print "Training accuracy =", classifier.score(trn_data, trn_labels)
print "Dev accuracy =", classifier.score(dev_data, dev_labels)


# prediction
tst_label=classifier.predict(tst_data)
print "Test results =", tst_label
fileObject = open('lr-test.pred', 'w')
for label in tst_label:
    fileObject.write(label)
    fileObject.write('\n')
fileObject.close()

