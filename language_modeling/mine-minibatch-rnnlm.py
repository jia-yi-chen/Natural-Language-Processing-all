## Author: Jiayi Chen 
## Time-stamp:  11/26/2018

import torch
import numpy as np
import os
import torch.nn as nn
import argparse
import time
from my_perplexity import avg_logprob_i, perplexity, repackage_hidden


parser = argparse.ArgumentParser(description='CS6501 NLP Project3 RNN Language Modeling')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size')
args = parser.parse_args()



# extract vocabulary, change token to vector, , and batchify
def pre_processing_trndata_withpad(path, mod):
    assert os.path.exists(path)
    # Add words to the dictionary
    total_tokennum = 0
    dictionary = ['<pad>']
    dictionary_idx = {}
    dictionary_idx['<pad>']=0
    sequencelen_list = []
    if mod=='from_source':
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
        Ns = len(sequencelen_list)

        fileObject = open('vocabulary_saved_withpad.txt', 'w')
        for v in range(V):
            fileObject.write(dictionary[v])
            fileObject.write(' ')
            fileObject.write(str(dictionary_idx[dictionary[v]]))
            if v<V-1:
                fileObject.write('\n')
        fileObject.close()

    elif mod=='from_savedfile':
        voc_file = open("vocabulary_saved_withpad.txt", "r")
        for line in voc_file:
            voc_and_index = line.split()
            dictionary.append(voc_and_index[0])
            dictionary_idx[voc_and_index[0]]=int(voc_and_index[1])
        voc_file.close()

        with open(path, 'r', encoding="utf8") as f:
            for sentence in f:
                words = sentence.split()
                total_tokennum += len(words)
                sequencelen_list.append(len(words))
        Ns=len(sequencelen_list)
        V=len(dictionary)


    # Tokenize file content
    with open(path, 'r', encoding="utf8") as f:
        # change each word-token to the V-dim vector
        # if mini_batch ==1:
        hot_data = torch.LongTensor(total_tokennum)# dim=1,800,340  标量指示index
        sequencelen_listtensor=torch.LongTensor(Ns)
        idx=0
        idxs=0
        for line in f:
            words = line.split()
            for word in words:
                hot_data[idx]=dictionary_idx[word]
                idx+=1
            sequencelen_listtensor[idxs]=len(words)
            idxs += 1


        # sequence_list=[]
        # for line in f:
        #     sentencelist=[]
        #     words = line.split()
        #     for word in words:
        #         sentencelist.append(dictionary_idx[word])
        #     sequence_list.append(sentencelist)


    return dictionary,dictionary_idx,  V, sequencelen_listtensor,hot_data,total_tokennum,len(sequencelen_list)


# Read data, and batchify
def pre_processing_tstdev_data(path,dictionary_idx):
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
        Ns=len(sequencelen_list)

    with open(path, 'r', encoding="utf8") as f:
        # change each word-token to the V-dim vector
        hot_data = torch.LongTensor(total_tokennum)
        sequencelen_listtensor=torch.LongTensor(Ns)
        idx=0
        idxs=0
        for line in f:
            words = line.split()
            for word in words:
                hot_data[idx]=dictionary_idx[word]
                idx+=1
            sequencelen_listtensor[idxs]=len(words)
            idxs += 1
    return sequencelen_listtensor, hot_data,total_tokennum,len(sequencelen_list)

# reture Tensor
device=torch.device("cuda")
def padding_seqs(data, seq_lenths, N, max_sequence_len):
    seq_tensor = torch.zeros((max_sequence_len,N)).long().cuda()
    index=0
    for idx, seqlen in enumerate(seq_lenths):
        cutsize=min(seqlen.item(),max_sequence_len)
        seq_tensor[:cutsize,idx] = torch.LongTensor(data[index:index+cutsize])# each 列
        index+=seqlen
    return seq_tensor


class RNN_LanguageModel(nn.Module):

    def __init__(self, mini_batch, max_sequence_length, Voc_num, inputembedding_sz, hidden_sz, lstm_layernum=1, dropout=0.0):

        super(RNN_LanguageModel, self).__init__()#继承nn.Module

        self.num_voc = Voc_num
        self.Ki = inputembedding_sz
        self.Kh = hidden_sz
        self.num_layer = lstm_layernum
        self.batch_size=mini_batch
        self.sequence_len=max_sequence_length


        # self.drop = nn.Dropout(dropout)

        # 3 layers
        self.wordembedding = nn.Embedding(self.num_voc, self.Ki )
        self.lstmlayer = getattr(nn, 'LSTM')(self.Ki , self.Kh , self.num_layer,dropout=dropout)
        self.decoder = nn.Linear(self.Kh, self.num_voc)


    def init_hidden(self,batch_size):
        weight = next(self.parameters())
        # (1,1,32)
        return (weight.new_zeros(self.num_layer, batch_size, self.Kh),
                weight.new_zeros(self.num_layer, batch_size, self.Kh))


    def forward(self, padded_input, hidden, list_of_lengths):

        # emb = self.drop(self.wordembedding(input))  # input(word_i(index_in_voc), batch_i)    emb(seq, batch, featuresz)
        emb=self.wordembedding(padded_input)

        embedded = nn.utils.rnn.pack_padded_sequence(emb, list_of_lengths)  # packed sequence

        # h(n)=g(W*input(n)+b, W*h(n-1)+b)
        # output(sequence_len, batch, featuresz)  [sequence_len, 1, 32]
        packed_output, hidden = self.lstmlayer(embedded, hidden)  # hidden(1, batch, 32)
        # output = self.drop(output)

        padded_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_output)

        # pred(n+1)=W*h(n)+b     [sequence_len , 27767]
        outputs = self.decoder(padded_outputs.view(padded_outputs.size(0) * padded_outputs.size(1),
                                                   padded_outputs.size(2)))  # predicted(seq*batch, featuresz)


        # [sequence_len, 1, 27767]
        return outputs.view(padded_outputs.size(0), padded_outputs.size(1), outputs.size(1)) , hidden  # predicted(seq, batch, V)



# 定义函数：截断反向传播(每个batch)
def detach_hiddens(hiddens):
    return [hidden.detach() for hidden in hiddens]





print('======== Load Data into vector:[token<->index], get Dictionary and apply =========')
# ------> dictionary_idx['<pad>']=0
dictionary,dictionary_idx, voc_size, trnseq_lengths, trn_hotdata, Wtrn, Ntrn=pre_processing_trndata_withpad("trn-wiki.txt",'from_savedfile')
devseq_lengths, dev_hotdata, Wdev, Ndev=pre_processing_tstdev_data("dev-wiki.txt",dictionary_idx)
tstseq_lengths, tst_hotdata, Wtst, Ntst=pre_processing_tstdev_data("tst-wiki.txt",dictionary_idx)
print('trn has {:3d} words and {:3d} sentences'.format(Wtrn, Ntrn))
print('dev has {:3d} words and {:3d} sentences'.format(Wdev, Ndev))
print('tst has {:3d} words and {:3d} sentences'.format(Wtst, Ntst))
print('Over!\n')


# print('======== Get Length of each sequence and get Sorting index (Tensor type) =========')
# trnseq_lengths = torch.LongTensor([sequencelen_list for sequencelen_list in trn_sequencelen_list]).cuda()
# devseq_lengths = torch.LongTensor([sequencelen_list for sequencelen_list in dev_sequencelen_list]).cuda()
# tstseq_lengths = torch.LongTensor([sequencelen_list for sequencelen_list in tst_sequencelen_list]).cuda()


print('======== Load Data into vector [ max_sequence_len, N] =========')
max_sequence_len=150
trn_data=padding_seqs(trn_hotdata, trnseq_lengths, Ntrn, max_sequence_len)
dev_data=padding_seqs(dev_hotdata, devseq_lengths, Ndev, max_sequence_len)
tst_data=padding_seqs(tst_hotdata, tstseq_lengths, Ntst, max_sequence_len)
print('Over!\n')



print('========  Data are Sorted by length in a decreasing order =========')
tsrseq_lengths, trn_idx = trnseq_lengths.sort(0, descending=True)
devseq_lengths, dev_idx = devseq_lengths.sort(0, descending=True)
tstseq_lengths, tst_idx = tstseq_lengths.sort(0, descending=True)
trn_data_sorted = trn_data.transpose(0,1)[trn_idx]# [L=max_sequence_len, N 句子个数]
dev_data_sorted = dev_data.transpose(0,1)[dev_idx]# [L, N]
tst_data_sorted = tst_data.transpose(0,1)[tst_idx]# [L, N]


print('======== Define the Model =========')
embed_sz=32
hidden_sz=32
LSTM_layernum=1

mini_batch=args.batch_size
RNN_LM = RNN_LanguageModel(mini_batch, max_sequence_len, voc_size, embed_sz, hidden_sz, LSTM_layernum, 0.0).to(device)
print('Over!\n')


print('======== Define Loss and Optimizer =========')
learning_rate=0.7
lossCE = nn.CrossEntropyLoss()# nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction), so no need softmax
optimization='SGD'
optimizer = torch.optim.SGD(RNN_LM.parameters(), lr=learning_rate)
print('Over!\n')


print('======== Start Training =========')
num_epochs=45
clipping_param=3
old_devperplexity=30000
for epoch in range(1, num_epochs+1):
    # epoch_start_time = time.time()
    # Turn on training mode which enables dropout.
    RNN_LM.train()

    start_time = time.time()
    ntokens = len(dictionary)

    # initiate h & c value to 0
    hiddens = RNN_LM.init_hidden(RNN_LM.batch_size)

    total_log_probs=0.
    total_loss=0.


    index=0
    for b in range(Ntrn//mini_batch):
        # LongTensor[a:b]  from a line to b line
        batchdata = trn_data_sorted[mini_batch*b : mini_batch*(b+1)].transpose(1,0).contiguous()# 2D [sequence_len, 1]
        padded_inputs=batchdata[:max_sequence_len-1]
        padded_targets = batchdata[1:max_sequence_len].view(-1)# contiguous() should be used before that
        list_of_lengths=tsrseq_lengths[mini_batch*b : mini_batch*(b+1)]
    # for i, seqence_len in enumerate(trn_sequencelen_list):
    #
    #     # every token in a sentence are read to RNN LM to compute a hidden
    #     inputs = trn_databatches[index : (index+seqence_len-1)]# 2D [sequence_len, 1]
    #     targets = trn_databatches[(index+1) : (index+seqence_len)].view(-1)# 1D [sequence_len*1]


        RNN_LM.zero_grad()


        ############################## Forward (calculate all words in sequence and minibatch=1 sentences) output: (seq, batch, V) ########################
        hiddens = detach_hiddens(hiddens)
        padded_outputs, hiddens = RNN_LM(padded_inputs, hiddens, list_of_lengths)# [sequence_len, 1=batchsz, 27767]
        padded_outputs=padded_outputs.view(-1, voc_size)# [sequence_len(*batchsz), 27767]


        ############################## loss.item() ##############3
        # -sum(yi*log(softmax(output_i))) *1/Ni 一个句子的
        loss = lossCE(padded_outputs, padded_targets) # include softmax part
        total_loss += loss.item()


        ########################## Backword (BBTT) ##################
        loss.backward()



        # Prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(RNN_LM.parameters(), clipping_param)
        # for p in RNN_LM.parameters():
        #     p.data.add_(-learning_rate, p.grad.data)

        optimizer.step()

        hiddens = repackage_hidden(hiddens)


        ppl=np.exp(loss.item())

        # 1/N * loglikelihood_sum
        # avg_lprb_i=avg_logprob_i(outputs,targets).item()
        #

        # # loglikelihood_sum (all sentences) = -sum(sum(yi*log(softmax(output_i))))
        # total_log_probs += avg_lprb_i * (seqence_len-1) # loss.item() = -sum(yi*log(softmax(output_i))) *1/Ni 一个句子的
        # index+=seqence_len # move to next sentence
        #
        if b % 300 == 0:
            # cur_avgloss = total_loss / (len(trn_sequencelen_list[0:i+1]))
            # cur_avg_logprobs = total_log_probs / (sum(trn_sequencelen_list[0:i+1])-len(trn_sequencelen_list[0:i+1]))
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {}/{} sentences | lr {} | cur_avg_loss {:5.2f} | current_perplexity {:5.2f}'.format(
                    epoch, b, Ntrn//mini_batch, learning_rate,    loss.item(),   ppl))
            start_time = time.time()
        perplexity_trn = perplexity(RNN_LM, trn_data_sorted, dictionary, dictionary_idx, trnseq_lengths,
                                    'only_calculate', device)
        print('End of epoch {} |  Perplexity number of training data is {:8.2f}\n'.format(epoch, perplexity_trn))
        perplexity_dev = perplexity(RNN_LM, dev_data_sorted, dictionary, dictionary_idx, devseq_lengths,
                                    'only_calculate', device)
        print('End of epoch {} |  Perplexity number of development data is {:8.2f}\n'.format(epoch, perplexity_dev))
        if perplexity_dev > old_devperplexity:
            print('Training Over!\n')
            break
        else:
            old_devperplexity = perplexity_dev
    print('Training Over!\n')

# print('Perplexity number of test data is {:8.2f}'.format(perplexity_tst))