## util.py
## Author: Author: CS 6501-005 NLP @ UVa
## Time-stamp: <yangfeng 10/14/2018 16:14:18>

import sys

class Sent(object):
    def __init__(self, tokens=None, tags=None):
        self.tokens = tokens
        self.tags = tags

def load_data(fname):
    print "Load data from {}".format(fname)
    data = []
    tokens, tags = [], []
    with open(fname) as fin:
        for line in fin:
            if line.startswith("TWEET") or line.startswith("TOKENS"):
                continue
            line = line.strip()
            if len(line) == 0:
                # create a new sentence
                sent = Sent(tokens, tags)
                data.append(sent)
                tokens, tags = [], []
            else:
                # attach to the current sentence
                items = line.split("\t")
                tokens.append(items[1])
                tags.append(items[0])
    if len(tokens) > 0:
        sent = Sent(tokens, tags)
        data.append(sent)
    print "Loaded {} sentences".format(len(data))
    return data

