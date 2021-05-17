import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import copy

class SynthesizerAttention(nn.Module):
    def __init__(self, n_embd,n_head,block_size,attn_pdrop,resid_pdrop):
        """
        
        n_embd : embedding size 
        n_head : number of attention heads
        block_size : length of seq
        attn_pdrop : attention dropout probability
        resid_pdrop : dropout prob after projection layer.
        
        """
        super().__init__()
        assert n_embd %n_head == 0
        self.w1 = nn.Linear(n_embd, n_embd)
        self.w2 = nn.Parameter(torch.zeros( n_embd //  n_head,
             block_size-1)) #d_k,T
        self.b2 = nn.Parameter(torch.zeros(block_size-1)) #T
        # value projection
        self.value = nn.Linear( n_embd,  n_embd) #dmodel,dmodel
        # regularization
        self.attn_drop = nn.Dropout( attn_pdrop)
        self.resid_drop = nn.Dropout( resid_pdrop)
        # output projection
        self.proj = nn.Linear( n_embd,  n_embd) #dmodel,dmodel
        # causal mask to ensure that attention is only applied to the left in
        #     the input sequence
        self.register_buffer("mask", torch.tril(
            torch.ones( block_size,  block_size)).view(
                1, 1,  block_size,  block_size)) #mask
        self.n_head =  n_head
        self.block_size =  block_size

        nn.init.uniform_(self.w2,-0.001,0.001)

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        # @ : The matrix multiplication(s) are done between the last two dimensions
        d_k = C//self.n_head
        relu_out = F.relu(self.w1(x)).\
            view(B,T,self.n_head,d_k).transpose(1,2)     

        v = self.value(x).view(B,T,self.n_head,d_k).transpose(1,2)   
        scores = (relu_out@self.w2)  + self.b2  
        
        scores = scores[:,:,:T,:T] # to ensure it runs for T<block_size
        scores = scores.masked_fill(self.mask[:,:,:T,:T]==0,-1e10)
        prob_attn = F.softmax(scores,dim=-1)
        y = prob_attn@v

        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.resid_drop(self.proj(y))
        return y