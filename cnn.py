#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    Class that implements the CNN architecture
    """
    def __init__(self, e_char, f, m_word, k=5):
        super(CNN,self).__init__()
        self.conv1d = nn.Conv1d(in_channels = e_char, out_channels = f, kernel_size=k)
        self.maxpool1d = nn.MaxPool1d(kernel_size = m_word - k + 1)
    def forward(self,x_reshaped):
        x = self.conv1d(x_reshaped)
        torch.relu_(x)
        x = self.maxpool1d(x)
        return x #.squeeze()

### END YOUR CODE

