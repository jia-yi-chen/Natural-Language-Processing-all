## POSTagging_HMM.py
## Author: Jiayi Chen 
## Time-stamp:  10/28/2018

from nltk import FreqDist
import numpy as np
import math

# Read words in test data 
def original(fname):
    file = open(fname, 'r')
    All_words=[]
    for line in file:
        Words = [w for w in line.split()]
        All_words.extend(Words)
    return All_words


# Read words + tags in training data 
def scan_data(fname):
    file = open(fname, 'r')
    All_words=[]
    All_tags=[]
    begin_index=[]
    end_index=[]
    for line in file:
        begin_index.append(len(All_tags))
        Words_Tags=[w for w in line.split()]
        end_index.append(len(All_tags)+len(Words_Tags)-1)
        Words=[wt.split("/")[0].lower() for wt in Words_Tags]
        Tags=[wt.split("/")[1] for wt in Words_Tags]
        All_words.extend(Words)
        All_tags.extend(Tags)
    return All_words,All_tags,begin_index,end_index


# Set the words with <K frequency into 'Unk' token 
def Preprocessing(Words,K):
    fdist = FreqDist(Words)
    vocabulary0 = fdist.keys()
    print 'Vocabulary size before pre-processing:', len(vocabulary0)
    Words2=Words
    for i in range(len(Words)):  # discard low-frequency words
        if fdist[Words[i]] < K:
            Words2[i]='Unk'
    return Words2


# When reading words + tags from development data 
# Setting words that are not in Vocab with 'Unk',
def scan_data_dev(fname,vocabulary):
    file = open(fname, 'r')
    All_words=[]
    All_tags=[]
    begin_index=[]
    end_index=[]
    for line in file:
        begin_index.append(len(All_tags))
        Words_Tags=[w for w in line.split()]
        end_index.append(len(All_tags)+len(Words_Tags)-1)
        Words=[]
        Tags=[]
        for wt in Words_Tags:
            w=wt.split("/")[0].lower()
            if w not in vocabulary:
                Words.append('Unk')
            else:
                Words.append(w)
            Tags.append(wt.split("/")[1])
        All_words.extend(Words)
        All_tags.extend(Tags)
    return All_words,All_tags,begin_index,end_index


# When reading words from test data,
# Setting words that are not in Vocab with 'Unk',
def scan_data_tst(fname,vocabulary):
    file = open(fname, 'r')
    All_words=[]
    All_tags=[]
    begin_index=[]
    end_index=[]
    for line in file:
        begin_index.append(len(All_words))
        Words=[w.lower() for w in line.split()]
        end_index.append(len(All_words)+len(Words)-1)
        Words_tokenize=[]
        for wt in Words:
            if wt not in vocabulary:
                Words_tokenize.append('Unk')
            else:
                Words_tokenize.append(wt)
        All_words.extend(Words_tokenize)
    return All_words,begin_index,end_index


# Answer for Section 1.2 Q2 and Q3
# Estimate transition and emission probabilities
def train(Words, Tags, begin_index, end_index, transition_filename, emission_filename):
    start='START'
    end='END'
    fdisk = FreqDist(Tags)
    POS = fdisk.keys() #  [A, C, D, M, N, O, P, R, V, W]
    Y=len(POS)
    L = Y+1

    """
    Transition Probability
    """
    transition = np.zeros([L,L])
    transition_dict={}

    # calculate P( A/C/D/M/N.. | START) = P( START , A/C/D/M/N.. ) / P(START)
    transition_dict['START']={}
    Count1=dict(zip(POS, [0.0] * Y))
    for indx in begin_index:
        tag = Tags[indx]
        Count1[tag]=Count1[tag]+1.0 # P( START , A/C/D/M/N.. )
    for j in range(Y):
        transition[0][j] = Count1[POS[j]] / len(begin_index)     # number of sentences = P(START)
        transition_dict['START'][POS[j]] = transition[0][j]
    transition_dict['START']['END'] = transition[0][L - 1]


    # Eg. P( A/C/D/M/N../END | A) = P( A/C/D/M/N../END , A) / P(A)
    #     P(A) is important
    # Count 'A' (or 'C' or ....) appearance times
    P_tags=dict(zip(POS, [0.0] * Y ))
    for tag in Tags:
        P_tags[tag]=P_tags[tag]+1.0



    # Count the apearance of every tags which appear after 'A'
    for i in range(Y):
        tag_prev=POS[i]
        transition_dict[tag_prev] = {}
        Count1_i=dict(zip(POS, [0.0] * Y )) # length is l-1
        Count1_i['END']=0.0
        current_sentence=0

        # Count all the sequential situations
        for indx in range(len(Tags)-1):
            # if 'A' is not the last word tag, count C(A , A/C/D/M/N..)
            if indx < end_index[current_sentence]:
                if Tags[indx] == tag_prev:
                    next = Tags[indx + 1]
                    Count1_i[next]=Count1_i[next]+1.0
            # if 'A' is the last word tag, count C(A , END)
            elif indx == end_index[current_sentence]:#
                if Tags[indx] == tag_prev:
                    Count1_i['END']=Count1_i['END']+1.0
                current_sentence=current_sentence+1  # move to next sentence

        # Bayesian
        # Eg.  Count('A', 'blablablab')  / Count('A')
        for j in range(Y):
            transition[i + 1][j] = Count1_i[POS[j]] / P_tags[tag_prev]
            transition_dict[tag_prev][POS[j]] = transition[i + 1][j]
        transition[i + 1][L-1] = Count1_i['END'] / P_tags[tag_prev]#
        transition_dict[tag_prev]['END'] = transition[i + 1][L - 1]


    # check for each yt-1,  sum(P(yt|yt-1))=1
    print "\nchecking for each yt-1,  sum(P(yt|yt-1))=1......"
    sum=0.0
    for j in range(Y):
        sum=sum+transition[0][j]
    print "sum up P( A/C/D/M/N.. | START) = ", sum
    sums=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,]
    for i in range(Y):
        sums[i] = 0.0
        for j in range(Y+1):
            sums[i] = sums[i] + transition[i + 1][j]
        print "sum up P( A/C/D/M/N/.../END | ", POS[i], ") = ", sums[i]
    print "checking over."


    # write transition probability to file
    print "\nwriting tprob.txt......"
    with open(transition_filename, "w") as f:
        # write P( A/C/D/M/N.. | START)
        for j in range(Y):
            f.write(', '.join([start,POS[j],str(transition[0][j])]))
            f.write("\n")
        # write P( A/C/D/M/N.. | A/C/D/M/N..)
        for i in range(Y):
            for j in range(Y):
                f.write(', '.join([POS[i], POS[j], str(transition[i+1][j])]))
                f.write("\n")
        # write P( END | A/C/D/M/N..)
        for i in range(Y):
            f.write(', '.join([POS[i], end , str(transition[i+1][L-1])]))
            f.write("\n")
    print "writing over."







    """
    Emission Probability
    """
    print "\n"
    fdist2 = FreqDist(Words)
    vocabulary = fdist2.keys()
    V=len(vocabulary)
    emission = {}

    for i in range(Y):
        tag = POS[i]
        Count2 = P_tags[tag]
        Count1 = dict(zip(vocabulary, [0.0] * V))  # length is l-1
        for indx in range(len(Words)):
            if Tags[indx] == tag:
                w = Words[indx]
                Count1[w]=Count1[w]+1.0

        emission_i = dict(zip(vocabulary, [0.0] * V))
        for voc in vocabulary:
            emission_i[voc]=Count1[voc]/Count2
        emission[tag]=emission_i

    # write emission probability to file
    print "\nwriting eprob.txt......"
    with open(emission_filename, "w") as f:
        # write P( A/C/D/M/N.. | A/C/D/M/N..)
        for i in range(Y):
            tag = POS[i]
            for voc in vocabulary:
                f.write(', '.join([tag, voc, str(emission[tag][voc])]))
                f.write("\n")
    print "writing over."

    # check for each yt,  sum(P(xt|yt))=1
    print "\nchecking for each yt,  sum(P(xt|yt))=1......."
    for i in range(Y):
        tag = POS[i]
        sum=0.0
        for voc in vocabulary:
            sum=sum+emission[tag][voc]
        print "sum up P(vocabularies|",tag,") =",sum
    print "checking over."

    return transition_dict, emission, vocabulary, V

def train_alphabeta(alpha, beta, Words, Tags, begin_index, end_index, transition_filename, emission_filename, write_or_not):
    start='START'
    end='END'
    fdisk = FreqDist(Tags)
    POS = fdisk.keys() #  [A, C, D, M, N, O, P, R, V, W]
    Y=len(POS)
    L = Y+1

    """
    Transition Probability
    """
    transition = np.zeros([L,L])
    transition_dict={}

    # calculate P( A/C/D/M/N.. | START) = P( START , A/C/D/M/N.. ) / P(START)
    transition_dict['START']={}
    Count1=dict(zip(POS, [0.0] * Y))
    for indx in begin_index:
        tag = Tags[indx]
        Count1[tag]=Count1[tag]+1.0 # P( START , A/C/D/M/N.. )
    for j in range(Y):
        transition[0][j] = (Count1[POS[j]]+ beta) / (len(begin_index) + L * beta)     # number of sentences = P(START)
        transition_dict['START'][POS[j]] = transition[0][j]
    transition_dict['START']['END'] = (transition[0][L - 1] + beta)/ (len(begin_index) + L * beta)


    # calculate  P(A) and P(C) ...........
    P_tags=dict(zip(POS, [0.0] * Y ))
    for tag in Tags:
        P_tags[tag]=P_tags[tag]+1.0

    # calculate P( A/C/D/M/N.. | A)  and  P( A/C/D/M/N.. | C) .........
    # Count the apearance of every tags which appear after 'A'
    for i in range(Y):
        tag_prev=POS[i]
        transition_dict[tag_prev] = {}
        Count1_i=dict(zip(POS, [0.0] * Y )) # length is l-1
        Count1_i['END']=0.0
        current_sentence=0

        # Count all the sequential situations
        for indx in range(len(Tags)-1):
            # if 'A' is not the last word tag, count C(A , A/C/D/M/N..)
            if indx < end_index[current_sentence]:
                if Tags[indx] == tag_prev:
                    next = Tags[indx + 1]
                    Count1_i[next]=Count1_i[next]+1.0
            # if 'A' is the last word tag, count C(A , END)
            elif indx == end_index[current_sentence]:#
                if Tags[indx] == tag_prev:
                    Count1_i['END']=Count1_i['END']+1.0
                current_sentence=current_sentence+1  # move to next sentence

        # Bayesian
        # Eg.  Count('A', 'blablablab')  / Count('A')
        for j in range(Y):
            transition[i + 1][j] = (Count1_i[POS[j]]+ beta) / (P_tags[tag_prev] + L*beta)
            transition_dict[tag_prev][POS[j]] = transition[i + 1][j]
        transition[i + 1][L-1] = (Count1_i['END']+beta) / (P_tags[tag_prev]+ L*beta)#
        transition_dict[tag_prev]['END'] = transition[i + 1][L - 1]

    if write_or_not==1:
        # write transition probability to file
        with open(transition_filename, "w") as f:
            # write P( A/C/D/M/N.. | START)
            for j in range(Y):
                f.write(', '.join([start,POS[j],str(transition[0][j])]))
                f.write("\n")
            # write P( A/C/D/M/N.. | A/C/D/M/N..)
            for i in range(Y):
                for j in range(Y):
                    f.write(', '.join([POS[i], POS[j], str(transition[i+1][j])]))
                    f.write("\n")
            # write P( END | A/C/D/M/N..)
            for i in range(Y):
                f.write(', '.join([POS[i], end , str(transition[i+1][L-1])]))
                f.write("\n")



    """
    Emission Probability
    """
    fdist2 = FreqDist(Words)
    vocabulary = fdist2.keys()
    V=len(vocabulary)
    emission = {}

    for i in range(Y):
        tag = POS[i]
        Count2 = P_tags[tag]
        Count1 = dict(zip(vocabulary, [0.0] * V))  # length is l-1
        for indx in range(len(Words)):
            if Tags[indx] == tag:
                w = Words[indx]
                Count1[w]=Count1[w]+1.0

        emission_i = dict(zip(vocabulary, [0.0] * V))
        for voc in vocabulary:
            emission_i[voc] = (Count1[voc] + alpha)/(Count2 + V*alpha)

        emission[tag] = emission_i

    if write_or_not==1:
        # write emission probability to file
        with open(emission_filename, "w") as f:
            # write P( A/C/D/M/N.. | A/C/D/M/N..)
            for i in range(Y):
                tag = POS[i]
                for voc in vocabulary:
                    f.write(', '.join([tag, voc, str(emission[tag][voc])]))
                    f.write("\n")

    return transition_dict, emission, vocabulary, V

class HMM():
    """
    Hidden Markov Model - for POS Tagging
    Author: Jiayi Chen
    """
    def __init__(self,transition,emission, vocabulary, V):
        self.transition = transition
        self.emission = emission
        self.num_states= len(emission)
        self.vocabulary=vocabulary
        self.V=V
        self.POS=self.emission.keys()
        print 'POS : ', self.POS

    # decode the hidden tags of a sentences
    def viterbi_decoder(self, sequence):
        l = len(sequence)
        Vm = np.empty([self.num_states, l])
        Bm = np.empty([self.num_states, l])

        # initiate  m=1
        x1=sequence[0]
        P_x1_if_y1 = [self.emission[pos][x1] for pos in self.POS]
        P_START_2_y1 = [self.transition['START'][pos] for pos in self.POS] # P( y1 | START )  :   P ( H | Start)  ,  P ( L | Start)
        for j in range(self.num_states):
            Vm[j, 0] = math.log(P_START_2_y1[j]) + math.log(P_x1_if_y1[j])  # log( S0(H) )
            Bm[j, 0] = 0

        # m={2...M}
        for i in range(1, l):
            xm = sequence[i]
            for j in range(0, self.num_states):
                tag_kprime=self.POS[j]

                V_pre = [Vm[k][i - 1] for k in range(self.num_states)]  # scan all the previous best value

                P_ypre_2_ym = [self.transition[pos][tag_kprime] for pos in self.POS]  # P( yi=j | all the previous state)
                P_xm_if_ym = self.emission[tag_kprime][xm]  # P( xi | H(L) )   =  P( x | yi = j )

                V_all = [ math.log(P_ypre_2_ym[k]) + math.log(P_xm_if_ym) + V_pre[k] for k in range(self.num_states)]
                Vm[j,i] = max(V_all)
                Bm[j,i] = np.argmax(V_all)

        # m=M+1  , calculate yM
        V_M = [ Vm[indx,l - 1] for indx in range(self.num_states) ]  # scan all the last value
        P_yM_2_END = [self.transition[pos]['END'] for pos in self.POS]  # P( yi=j | all the previous state)
        tags_M = np.argmax([math.log(P_yM_2_END[indx]) + V_M[indx] for indx in range(self.num_states)]) # P( yi=j | all the previous state) +  Vi-1(k)

        # decode hidden states y1:M
        results = np.empty(l, int)
        POS_results = []
        results[l - 1] = tags_M
        for j in range(l-1, 0, -1):  # j never == 0
            results[j - 1] = Bm[results[j], j]
        results=results.tolist()
        for indx in range(l):
            POS_results.append(self.POS[results[indx]])

        return POS_results
        

    # evaluate a sentence
    def score(self, Tags, Tags_predicted):
        sentence_l=len(Tags)
        count_correct_POS=0.0
        for i in range(sentence_l):
            if Tags[i]==Tags_predicted[i]:
                count_correct_POS=count_correct_POS+1.0
        return count_correct_POS, sentence_l



print "\nLoading Trn data .........................."
Words,Tags,begin_index,end_index = scan_data("trn.pos")
print 'number of words : ', len(Words)
print 'number of tags : ', len(Tags)



print "\nPre-processing Trn data .........................."
K=3
Words_new = Preprocessing(Words,K)
print 'K: ', K




print "\nTraining 'Transition & Emission Probabilities' from Trn data .........................."
# alpha,beta = 0.01, 1 #0.94988887166
alpha,beta = 0.005, 0.25 #0.949923
#  alpha,beta = 0.1, 0.1 #0.949595975259
# alpha,beta = 0.01, 1000 # 0.949423683258
# alpha,beta = 0.01, 10 #  0.94988887166
# alpha,beta = 0.001, 100 # 0.94988887166
# alpha ,beta = 1, 1  # 0.943608828242
# transition, emission , vocabulary, V  = train(Words_new, Tags, begin_index,end_index,"tprob.txt","eprob.txt")
transition, emission , vocabulary, V = train_alphabeta(alpha , beta, Words_new, Tags, begin_index,end_index,"tprob-smoothed.txt","eprob-smoothed.txt",0)
print 'Vocabulary size after pre-processing is: ', V




print "\nDefine Hidden Markov Model .........................."
hmm = HMM(transition, emission, vocabulary, V)



print "\nProcessing Dev data .........................."
Dev_Words, Dev_Tags, Dev_begin_index, Dev_end_index = scan_data_dev("dev.pos",vocabulary)
correct_tagging_num=0.0
total_tagging_num=0.0
for i in range(len(Dev_begin_index)):
    print "\n"
    print "the ", i+1, 'th Dev sentence'
    Dev_i_sentence=Dev_Words[Dev_begin_index[i]:Dev_end_index[i]+1]
    Dev_i_tags = Dev_Tags[Dev_begin_index[i]:Dev_end_index[i]+1]
    print '      ',Dev_i_sentence
    print '      Real_Tags : ',Dev_i_tags

    Y_i = hmm.viterbi_decoder(Dev_i_sentence)
    print '      Prediction: ', Y_i

    count_correct_POS, sentence_l = hmm.score(Dev_i_tags , Y_i)
    correct_tagging_num=correct_tagging_num+count_correct_POS
    total_tagging_num=total_tagging_num+sentence_l
accuracy = correct_tagging_num/total_tagging_num
print '\nThe accuracy of my decoder on the dev data is : ', accuracy
# Tune: 0.943 -(1,1)       0.946 - (0.5, 1)   0.94904 - (0.001, 10)   0.94915 - (0.001, 100)   0.948 - (0.001, 1000)
# 0.94916 - (0.0001, 100)
print "The end of processing Dev data !!!!!!!!!!!!!!!!!!!!"





print "\nTagging Test data .........................."
Tst_Words, Tst_begin_index, Tst_end_index = scan_data_tst("tst.word",vocabulary)# change-pre-processed words in sentence
Original_words = original("tst.word")  # original words in sentence
if alpha==1 and beta==1:
    output="viterbi.txt"
else:
    output="viterbi-tuned.txt"
with open(output, "w") as f:
    for i in range(len(Tst_begin_index)):
        print "\n"
        print "the ", i+1, 'th Test sentence'

        Test_i_sentence=Tst_Words[Tst_begin_index[i]:Tst_end_index[i]+1]#change- pre-processed words in sentence
        print '      ',Test_i_sentence

        Y_i = hmm.viterbi_decoder(Test_i_sentence)
        print '      Prediction: ', Y_i

        Original_sentence_i=Original_words[Tst_begin_index[i]:Tst_end_index[i]+1]
        for w in range(len(Test_i_sentence)):
            f.write('/'.join([Original_sentence_i[w], Y_i[w]]))
            if w==len(Test_i_sentence)-1:
                f.write("\n")
            else:
                f.write(' ')
print "\n"
print "The end of processing Test data !!!!!!!!!!!!!!!!!!!! already write to ",  output

