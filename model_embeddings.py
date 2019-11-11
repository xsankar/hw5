#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        self.embed_size = embed_size # needed for saving. But not mentioned anywhere else !
        e_char = 50
        drop_out = 0.3
        pad_token_idx = vocab.char2id['<pad>']
        self.embeddings = nn.Embedding(num_embeddings = len(vocab.char2id),
                                       embedding_dim = e_char, padding_idx = pad_token_idx)
        # print(CNN.__dict__)
        # took sometime to find the __init instead of __init__ error ! (11/11/19)
        self.cnn = CNN(e_char = e_char,f = embed_size,m_word = 21)
        self.highway = Highway(e_word = embed_size)
        self.dropout = nn.Dropout(drop_out)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        x = self.embeddings(input)
        # print("x-1",x.shape) # [10, 5, 21, 50]
        s_len, b_size,max_w_len,e_char = x.shape
        x = x.reshape(-1,e_char,max_w_len)
        x = self.cnn(x)
        x = x.reshape(s_len,b_size,-1)
        x = self.highway(x)
        x = self.dropout(x)
        return x
        ### END YOUR CODE

