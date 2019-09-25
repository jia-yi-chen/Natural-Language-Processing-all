## Author: Jiayi Chen 
## Time-stamp:  11/26/2018

import torch
import numpy as np
import os
import torch.nn as nn
import argparse
import math
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import time
from my_perplexity import avg_logprob_i, perplexity,  repackage_hidden


parser = argparse.ArgumentParser(description='CS6501 NLP Project3 RNN Language Modeling')
parser.add_argument('--optimizer', type=str, default='Adagrad',
                    help='type of optimizer')
args = parser.parse_args()




# extract vocabulary, change token to vector, , and batchify
def pre_processing_trndata(path, mod, mini_batch, cut_sequence_len):
    assert os.path.exists(path)
    # Add words to the dictionary
    total_tokennum = 0
    dictionary = []
    dictionary_idx = {}
    sequencelen_list = []
    if mod == 'from_source':
        with open(path, 'r', encoding="utf8") as f:
            for sentence in f:
                words = sentence.split()
                sequencelen_list.append(len(words))
                total_tokennum += len(words)
                for word in words:
                    if word not in dictionary:
                        dictionary.append(word)
                        dictionary_idx[word] = len(dictionary) - 1
        V = len(dictionary)

        fileObject = open('vocabulary_saved.txt', 'w')
        for v in range(V):
            fileObject.write(dictionary[v])
            fileObject.write(' ')
            fileObject.write(str(dictionary_idx[dictionary[v]]))
            if v < V - 1:
                fileObject.write('\n')
        fileObject.close()

    elif mod == 'from_savedfile':
        voc_file = open("vocabulary_saved.txt", "r")
        for line in voc_file:
            voc_and_index = line.split()
            dictionary.append(voc_and_index[0])
            dictionary_idx[voc_and_index[0]] = int(voc_and_index[1])
        voc_file.close()

        with open(path, 'r', encoding="utf8") as f:
            for sentence in f:
                words = sentence.split()
                total_tokennum += len(words)
                sequencelen_list.append(len(words))

        V = len(dictionary)

    # Tokenize file content
    with open(path, 'r', encoding="utf8") as f:
        # change each word-token to the V-dim vector
        if mini_batch == 1:
            hot_data = torch.LongTensor(total_tokennum)  # dim=1,800,340  标量指示index
            idx = 0
            for line in f:
                words = line.split()
                for word in words:
                    hot_data[idx] = dictionary_idx[word]
                    idx += 1

        # sequence_list=[]
        # for line in f:
        #     sentencelist=[]
        #     words = line.split()
        #     for word in words:
        #         sentencelist.append(dictionary_idx[word])
        #     sequence_list.append(sentencelist)

    return dictionary, dictionary_idx, V, sequencelen_list, hot_data, total_tokennum, len(sequencelen_list)


# Read data, and batchify
def pre_processing_tstdev_data(path, dictionary_idx, mini_batch, cut_sequence_len):
    assert os.path.exists(path)
    # Add words to the dictionary
    total_tokennum = 0
    sequencelen_list = []
    # Tokenize file content
    with open(path, 'r', encoding="utf8") as f:
        for sentence in f:
            words = sentence.split()
            total_tokennum += len(words)
            sequencelen_list.append(len(words))

    with open(path, 'r', encoding="utf8") as f:
        # change each word-token to the V-dim vector
        if mini_batch == 1:
            hot_data = torch.LongTensor(total_tokennum)
            idx = 0
            for line in f:
                words = line.split()
                for word in words:
                    if word not in dictionary:
                        hot_data[idx] = dictionary_idx['<unk>']
                    else:
                        hot_data[idx] = dictionary_idx[word]
                    idx += 1
    return sequencelen_list, hot_data, total_tokennum, len(sequencelen_list)


# reture Tensor
device = torch.device("cuda")


def data2batch(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


class RNN_LanguageModel(nn.Module):

    def __init__(self, mini_batch, Voc_num, inputembedding_sz, hidden_sz, lstm_layernum=1, dropout=0.0):
        super(RNN_LanguageModel, self).__init__()  # 继承nn.Module

        self.num_voc = Voc_num
        self.Ki = inputembedding_sz
        self.Kh = hidden_sz
        self.num_layer = lstm_layernum
        self.batch_size = mini_batch

        # 3 layers
        self.wordembedding = nn.Embedding(self.num_voc, self.Ki)
        self.lstmlayer = getattr(nn, 'LSTM')(self.Ki, self.Kh, self.num_layer, dropout=dropout)
        self.hidden2wordindex = nn.Linear(self.Kh, self.num_voc)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        # (1,1,32)
        return (weight.new_zeros(self.num_layer, batch_size, self.Kh),
                weight.new_zeros(self.num_layer, batch_size, self.Kh))

    def forward(self, input, hidden):
        # emb = self.drop(self.wordembedding(input))  # input(word_i(index_in_voc), batch_i)    emb(seq, batch, featuresz)
        emb = self.wordembedding(input)

        # h(n)=g(W*input(n)+b, W*h(n-1)+b)
        # output(sequence_len, batch, featuresz)  [sequence_len, 1, 32]
        output, hidden = self.lstmlayer(emb, hidden)  # hidden(1, batch, 32)
        # output = self.drop(output)

        # pred(n+1)=W*h(n)+b     [sequence_len , 27767]
        prediction_prob = self.hidden2wordindex(output.view(output.size(0) * output.size(1),
                                                            output.size(2)))  # predicted(seq*batch, featuresz)

        # [sequence_len, 1, 27767]
        return prediction_prob.view(output.size(0), output.size(1),
                                    prediction_prob.size(1)), hidden  # predicted(seq, batch, V)


# 截断反向传播(每个batch)
def detach_hiddens(hiddens):
    return [hidden.detach() for hidden in hiddens]


print('======== Load Data into vector:[token<->index], get Dictionary and apply =========')
CutPadding = False
cut_sequence_len = 50
mini_batch = 1
dictionary, dictionary_idx, voc_size, trn_sequencelen_list, trn_hotdata, Wtrn, Ntrn = pre_processing_trndata(
    "trn-wiki.txt", 'from_savedfile', mini_batch, cut_sequence_len)
dev_sequencelen_list, dev_hotdata, Wdev, Ndev = pre_processing_tstdev_data("dev-wiki.txt", dictionary_idx, mini_batch,
                                                                           cut_sequence_len)
tst_sequencelen_list, tst_hotdata, Wtst, Ntst = pre_processing_tstdev_data("tst-wiki.txt", dictionary_idx, mini_batch,
                                                                           cut_sequence_len)
print('trn has {:3d} words and {:3d} sentences'.format(Wtrn, Ntrn))
print('dev has {:3d} words and {:3d} sentences'.format(Wdev, Ndev))
print('tst has {:3d} words and {:3d} sentences'.format(Wtst, Ntst))
print('Over!\n')

print('======== Load Data into Batches =========')
mini_batch = 1
trn_databatches = data2batch(trn_hotdata, mini_batch)
mini_batch_foreval = 1
dev_databatches = data2batch(dev_hotdata, mini_batch_foreval)
tst_databatches = data2batch(tst_hotdata, mini_batch_foreval)
print('Over!\n')

print('======== Define the Model =========')
embed_sz = 32
hidden_sz = 32
LSTM_layernum = 1
RNN_LM = RNN_LanguageModel(mini_batch, voc_size, embed_sz, hidden_sz, LSTM_layernum, 0.0).to(device)
print('Over!\n')

print('======== Define Loss and Optimizer =========')
lossCE = nn.CrossEntropyLoss()  # nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction), so no need softmax
learning_rate = 0.7
if args.optimizer == 'SGDmomentum':
    optimizer = torch.optim.SGD(RNN_LM.parameters(), lr=learning_rate, momentum=0.9)
if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(RNN_LM.parameters())
if args.optimizer == 'Adagrad':
    optimizer = torch.optim.Adagrad(RNN_LM.parameters())
if args.optimizer == 'Adamax':
    optimizer = torch.optim.Adamax(RNN_LM.parameters())
if args.optimizer == 'SparseAdam':
    optimizer = torch.optim.SparseAdam(RNN_LM.parameters())
else:
    optimizer = torch.optim.Adagrad(RNN_LM.parameters())
optselected = args.optimizer
print(optselected)
print('Over!\n')

print('======== Start Training =========')
max_epochs = 40
clipping_param = 5
old_devperplexity = 30000
for epoch in range(1, max_epochs + 1):
    # epoch_start_time = time.time()
    # Turn on training mode which enables dropout.
    RNN_LM.train()

    start_time = time.time()
    ntokens = len(dictionary)

    # initiate h & c value to 0
    hiddens = RNN_LM.init_hidden(RNN_LM.batch_size)

    total_log_probs = 0.
    total_loss = 0.

    index = 0
    for i, seqence_len in enumerate(trn_sequencelen_list):

        # every token in a sentence are read to RNN LM to compute a hidden
        inputs = trn_databatches[index: (index + seqence_len - 1)]  # 2D [sequence_len, 1]
        targets = trn_databatches[(index + 1): (index + seqence_len)].view(-1)  # 1D [sequence_len*1]

        RNN_LM.zero_grad()

        ############################## Forward (calculate all words in sequence and minibatch=1 sentences) output: (seq, batch, V) ########################
        hiddens = detach_hiddens(hiddens)
        outputs, hiddens = RNN_LM(inputs, hiddens)  # [sequence_len, 1=batchsz, 27767]
        outputs = outputs.view(-1, voc_size)  # [sequence_len(*batchsz), 27767]

        ############################## loss.item() ##############3
        # -sum(yi*log(softmax(output_i))) *1/Ni 一个句子的
        loss = lossCE(outputs, targets)  # include softmax part
        total_loss += loss.item()

        ########################## Backword (BBTT) ##################
        loss.backward()

        # Prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(RNN_LM.parameters(), clipping_param)

        optimizer.step()

        hiddens = repackage_hidden(hiddens)

        # 1/N * loglikelihood_sum
        avg_lprb_i = avg_logprob_i(outputs, targets, device).item()

        # loglikelihood_sum (all sentences) = -sum(sum(yi*log(softmax(output_i))))
        total_log_probs += avg_lprb_i * (seqence_len - 1)  # loss.item() = -sum(yi*log(softmax(output_i))) *1/Ni 一个句子的
        index += seqence_len  # move to next sentence

        if i % 3000 == 0:
            cur_avgloss = total_loss / (len(trn_sequencelen_list[0:i + 1]))
            cur_avg_logprobs = total_log_probs / (
                        sum(trn_sequencelen_list[0:i + 1]) - len(trn_sequencelen_list[0:i + 1]))
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {}/{} sentences | lr {} | cur_avg_loss {:5.2f} | current_perplexity {:5.2f}'.format(
                epoch, i, len(trn_sequencelen_list), learning_rate, cur_avgloss, np.exp(-cur_avg_logprobs)))
            start_time = time.time()


    perplexity_trn = perplexity(RNN_LM, trn_databatches, dictionary, dictionary_idx, trn_sequencelen_list,
                                           'only_calculate',device)


    print('End of epoch {} |  Perplexity number of training data is {:8.2f}\n'.format(epoch, perplexity_trn))
    perplexity_dev = perplexity(RNN_LM, dev_databatches, dictionary, dictionary_idx, dev_sequencelen_list,
                                           'only_calculate', device)
    print('End of epoch {} |  Perplexity number of development data is {:8.2f}\n'.format(epoch, perplexity_dev))
    if perplexity_dev > old_devperplexity:
        print('Training Over!\n')
        # break   #do not stop
    else:
        old_devperplexity = perplexity_dev
print('Training Over!\n')



print('The optimizer is {}'.format(optselected))