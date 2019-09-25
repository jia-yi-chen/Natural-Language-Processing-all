#!/usr/bin/env python
#coding:utf-8
## Author: Jiayi Chen 
## Time-stamp:  09/22/2018


from my_bag_of_words import list_to_dict,read_from_file,setup_feature_vector_from_bag,buid_or_read_voc,bag_of_words_all
import numpy as np
import matplotlib.pyplot as plt
from time import time

class AveragedPerceptronClassifier():
  def __init__(self,learning_rate=1.0, num_epochs=50,trn_x_file='',trn_y_file='',dev_x_file='',dev_y_file=''):
    self.learning_rate=learning_rate
    self.num_epochs=num_epochs
    self.trainingset_X=read_from_file(trn_x_file)
    self.trainingset_y=read_from_file(trn_y_file)
    self.developSet_X=read_from_file(dev_x_file)
    self.developSet_y=read_from_file(dev_y_file)
    self.Testingset_X=read_from_file('tst.data')
    self.voc, self.n_voc = buid_or_read_voc(self.trainingset_X)# build vocabulary
    self.n_feature=2*self.n_voc+1
    self.n_label=2
    self.n_sample=len(self.trainingset_X)
    self.weights=np.ones(self.n_feature,float)#initialize weights
    self.accuracy_trn=[0]
    self.accuracy_dev = [0]
    self.vocabulary_dict = list_to_dict(self.voc)


    self.n_sample=30000
    self.X = bag_of_words_all(self.trainingset_X,self.n_sample,self.vocabulary_dict)

    self.Bag_DevSet = bag_of_words_all(self.developSet_X, len(self.developSet_X), self.vocabulary_dict)
    self.Bag_TstSet = bag_of_words_all(self.Testingset_X, len(self.Testingset_X), self.vocabulary_dict)



  def train(self):
    w=self.weights
    sum_w_count = 0
    sum_w = 0.0
    for epoch in range(self.num_epochs):
      n_errors = 0 # accumulate number of errors in this epoch
      count=0.0
      rate=0.0
      print '\n update', epoch, 'th ......................\n'
      for i in range(self.n_sample):
        if (i+1)%100==0:
          print '       training....... ',100.0*float(i)/30000.0, '% data'
        bag_x=self.X[i,:]
        real_y=self.trainingset_y[i]
        y_prediction,real_feature,unreal_feature=self.predict3(w,bag_x,real_y)
        if y_prediction!=real_y:
            w+=self.learning_rate*(real_feature-unreal_feature)
        else:
            w+=0.0
            count += 1.0
            rate = count/(i+1)
            sum_w += w
            sum_w_count += 1

      w = sum_w / sum_w_count
      self.accuracy_trn.append(self.compute_TrainingSet_accuracy(w))
      print 'accuracy in training set=',self.accuracy_trn, 'of ', epoch, 'th epoch'
      self.accuracy_dev.append(self.compute_DevelopmentSet_accuracy(w))
      print 'accuracy in development set=',self.accuracy_dev, 'of ', epoch, 'th epoch'
    self.weights=w
    return w

  def compute_TrainingSet_accuracy(self,w):
    correct = 0.0
    for i in range(self.n_sample):
      y_prediction = self.predict4(w, self.X[i,:])
      if self.trainingset_y[i] == y_prediction:
        correct += 1.0
    return correct / self.n_sample #only used for testing code



  def compute_DevelopmentSet_accuracy(self,w):
    correct = 0.0
    for i in range(self.Bag_DevSet.shape[0]):
      if (i + 1) % 100 == 0:
        print '       predict (development data) processing...... ', 100.0 * float(i) /len(self.developSet_X), '% data'
      y_prediction = self.predict4(w, self.Bag_DevSet[i,:])
      if self.developSet_y[i] == y_prediction:
        correct += 1.0
    return correct / len(self.developSet_X) #only used for testing code



  def test(self,tst_x_file):
    Testingset_X = read_from_file(tst_x_file)
    fileObject = open('averaged-perceptron-test.pred', 'w')
    for i in range(self.Bag_TstSet.shape[0]):
      if (i + 1) % 100 == 0:
        print '       predict (testing data) processing...... ', 100.0 * float(i) / len(Testingset_X), '% data'
      y_prediction = self.predict4(self.weights, self.Bag_TstSet[i,:])
      fileObject.write(y_prediction)
      fileObject.write('\n')
    fileObject.close()


  def predict3(self,w,bag_x,real_y):
    real_feature=''
    unreal_feature=''
    value_y0_feature=setup_feature_vector_from_bag(self.voc, bag_x, '0')
    value_y1_feature = setup_feature_vector_from_bag(self.voc, bag_x, '1')
    if real_y=='1':
        real_feature=value_y1_feature
        unreal_feature = value_y0_feature
    if real_y == '0':
        real_feature = value_y0_feature
        unreal_feature = value_y1_feature

    value_y0 = np.dot(w,value_y0_feature)
    value_y1 = np.dot(w,value_y1_feature)
    y_predicted=''
    if value_y0>value_y1:
        y_predicted='0'
    elif value_y0 <= value_y1:
        y_predicted='1'
    return y_predicted,real_feature,unreal_feature



  def predict4(self, w, bag_x):
    value_y0_feature = setup_feature_vector_from_bag(self.voc, bag_x, '0')
    value_y1_feature = setup_feature_vector_from_bag(self.voc, bag_x, '1')
    value_y0 = np.dot(w, value_y0_feature)
    value_y1 = np.dot(w, value_y1_feature)
    y_predicted=''
    if value_y0 > value_y1:
      y_predicted = '0'
    elif value_y0 <= value_y1:
      y_predicted = '1'
    return y_predicted





clf=AveragedPerceptronClassifier(1.0,6,'trn.data','trn.label','dev.data','dev.label')
t0 = time()
weights=clf.train()
print '\n'
print 'training is over!'
print 'Trained W:', weights
print 'Error rate on training set', 1.0-clf.accuracy_trn[-1]
print 'Error rate on development set', 1.0-clf.accuracy_dev[-1]
print 'training time is : ', time()-t0

fig=plt.figure(1)
fig1=fig.add_subplot(111)
fig1.set_title('Averaged Perceptron Training Iteration')
fig1.set_xlabel("Epochs")
plt.xlim(0,6.0)
fig1.set_ylabel("accuracies")
plt.ylim(0,1.5)
plot1 = fig1.plot(range(len(clf.accuracy_trn)), clf.accuracy_trn, '-r',label='accuracy curves on training set')
plot2 = fig1.plot(range(len(clf.accuracy_trn)), clf.accuracy_dev, '-b',label='accuracy curves on development set')
plt.legend(loc='upper right')
plt.show()
fig.savefig('Averaged_Perceptron_Plot.jpg',dpi=fig.dpi)





t1 = time()
clf.test('tst.data')
print 'testing time is : ', time()-t1

