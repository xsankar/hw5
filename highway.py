#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d
import torch
import torch.nn as nn

class Highway(nn.Module):
    def __init__(self,e_word): # drop_out=0.3
        super(Highway,self).__init__()
        self.proj = nn.Linear(e_word,e_word)
        self.gate = nn.Linear(e_word,e_word)
        # self.dropout = nn.Dropout(drop_out) # Not here, but in embeddings

    def forward(self,x_conv_out:torch.Tensor) -> torch.Tensor:
        # print("x_conv_out : ",x_conv_out.size())
        x_proj = torch.relu(self.proj(x_conv_out))
        # print("x_proj : ", x_proj.size())
        x_gate = torch.sigmoid(self.gate(x_conv_out))
        # print("x_gate : ", x_gate.size())

        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        # print("x_highway : ", x_highway.size())
        # Not here
        # x_highway = self.dropout(x_highway)
        return x_highway
### END YOUR CODE 

