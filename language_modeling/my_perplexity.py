## Author: Jiayi Chen 
## Time-stamp:  11/26/2018

import torch
import numpy as np
import os
import torch.nn as nn
import math
import torch.nn.functional as F
# import numpy as np


# device=torch.device("cuda")
def avg_logprob_i(outputs, targets,device):
    logsoftmax=F.log_softmax(outputs,1).to(device)

    log_likelihood = - F.nll_loss(logsoftmax, targets).to(device)# The negative log likelihood loss (logsoftmax---[sequence_length, voc_size])

    # neg_log_likelihood ----- already "/N"  sum(logs)/sequence_length
    return log_likelihood




def expavg(log_likelihood_sum, trn_sequencelen_list):
    # trn_sequencelen_list :  [N1+2 , N2+2, ........Nn+2]
    # N= (1+N1) + (1+N2) + ........
    N=sum(trn_sequencelen_list) - len(trn_sequencelen_list)
    return np.exp(-1/N*log_likelihood_sum)




def perplexity(model,data_batches, dictionary, dictionary_idx,sequencelen_list, mode,device):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if mode == 'writetofile':
        file_name = 'tst-logprob.txt'
        fileObject = open(file_name, 'w')
    total_log_probs = 0.
    hiddens = model.init_hidden(data_batches.size(1))
    index=0
    with torch.no_grad():
        for i, seqence_len in enumerate(sequencelen_list):
            # every token in a sentence are read to RNN LM to compute a hidden
            inputs = data_batches[index: (index + seqence_len - 1)]  # 2D [sequence_len, 1]
            targets = data_batches[(index + 1): (index + seqence_len)].view(-1)  # 1D [sequence_len*1]

            outputs, hiddens = model(inputs, hiddens)
            outputs = outputs.view(-1, model.num_voc)#p(wn|wn−1,…,w1,START)=softmax(Whn−1+b), where W is the weight matrix, hn−1 is the last hidden state from RNN and b is the bias term.
            avg_lprb_i = avg_logprob_i(outputs, targets,device)

            # loglikelihood_sum (all sentences) = -sum(sum(yi*log(softmax(output_i))))
            total_log_probs += avg_lprb_i * (seqence_len - 1)  # loss.item() = -sum(yi*log(softmax(output_i))) *1/Ni 一个句子的
            index += seqence_len  # move to next sentence

            hiddens = repackage_hidden(hiddens)
            if mode=='only_calculate':
                pass
            if mode=='writetofile':
                logprobs=F.log_softmax(outputs, 1).to(device)
                for w in range(seqence_len-1):
                    real_token_index=targets[w]
                    token=dictionary[real_token_index]
                    log_probability=logprobs[w,dictionary_idx[token]].item()
                    fileObject.write(token)
                    fileObject.write('\t')
                    fileObject.write(str(log_probability))
                    fileObject.write('\n')
    if mode == 'writetofile':
        fileObject.close()

    perplexitynumber = expavg(total_log_probs, sequencelen_list)
    return perplexitynumber





def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()#当反向传播经过这个node时，梯度就不会从这个node往前面传播
    else:
        return tuple(repackage_hidden(v) for v in h)