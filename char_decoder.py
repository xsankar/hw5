#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder,self).__init__()
        self.charDecoder = nn.LSTM(input_size = char_embedding_size,hidden_size = hidden_size)
        self.char_output_projection = nn.Linear(in_features = hidden_size, out_features = len(target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(num_embeddings = len(target_vocab.char2id),
                                           embedding_dim = char_embedding_size,
                                           padding_idx = target_vocab.char2id['<pad>'])
        self.target_vocab = target_vocab
        self.padding_ind = self.target_vocab.char2id['<pad>']
        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        char_embed = self.decoderCharEmb(input)
        out,(h_n,c_n) = self.charDecoder(char_embed,dec_hidden)
        scores = self.char_output_projection(out)
        return scores,(h_n,c_n)
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        scores, (h_n, c_n) = self.forward(char_sequence[:-1],dec_hidden) # Strip <END>
        loss = nn.CrossEntropyLoss(ignore_index=-self.padding_ind , reduction='sum')
        # print("scores-before",scores.shape) # [3, 5, 30] for sanity check
        embed_size, batch_size, y = scores.shape
        scores = scores.permute(1, 2, 0) # scores.reshape(-1,y)
        # print("scores-after",scores.shape) # [5, 30, 3]for sanity check
        target = char_sequence[1:].transpose(1, 0)
        # print("target",target.shape) # [5, 3] for sanity check
        return loss(scores,target)
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        output_words = []
        st_ind = self.target_vocab.start_of_word
        en_ind = self.target_vocab.end_of_word
        # print("InitialStates",initialStates[0].shape) # [1, 5, 3]
        batch_size = initialStates[0].shape[1]
        current_chars = torch.tensor([[st_ind] * batch_size],device=device)
        # complete all max length steps of the for-loop
        int_lstm_state = initialStates # h_t,c_t
        for _ in range(max_length):
            scores,int_lstm_state = self.forward(current_chars,int_lstm_state)
            current_chars = scores.argmax(-1)
            output_words += [current_chars]
        # truncate the output words afterwards and decode
        output_list = torch.cat(output_words).t().tolist()
        final_output=[]
        for wrd in output_list:
            the_word=""
            for id in wrd:
                if id == en_ind:
                    break
                else:
                    the_word += self.target_vocab.id2char[id]
            final_output += [the_word]
        return final_output
        ### END YOUR CODE

