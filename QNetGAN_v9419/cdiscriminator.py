import torch
import torch.nn as nn
import numpy as np

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        '''
        LSTM Cell Class in file qnet_gan_disc.py
        Instantiates an LSTM Cell
        
        @params
        input_size: int --> the size of the input
        hidden_size: int --> the size of the hidden layer
        '''
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = nn.Linear(input_size+hidden_size, 4 * hidden_size, bias=True)
        torch.nn.init.xavier_uniform_(self.cell.weight)
        torch.nn.init.zeros_(self.cell.bias)

    def forward(self, x, hidden):
        '''
        forward method in class LSTMCell
        you already kknow what this does
        '''
        hx, cx = hidden
        gates = torch.cat((x, hx), dim=1)
        gates = self.cell(gates)

        ingate, cellgate, forgetgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(torch.add(forgetgate, 1.0))
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, torch.tanh(cy))
        return (hy, cy)

#SHOULD WE MAKE A QUANTUM DISCRIMINATOR TOO
#AFTER WE MAKE QNETGAN WORK?
class Discriminator(nn.Module):
    def __init__(self, H_inputs, H, N, rw_len):
        '''
            H_inputs: input dimension
            H:        hidden dimension
            N:        number of nodes (needed for the up and down projection)
            rw_len:   number of LSTM cells
        '''
        super(Discriminator, self).__init__()
        self.W_down = nn.Linear(N, H_inputs, bias=False).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.W_down.weight)
        self.lstmcell = LSTMCell(H_inputs, H).type(torch.float64)
        self.lin_out = nn.Linear(H, 1, bias=True).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.lin_out.weight)
        torch.nn.init.zeros_(self.lin_out.bias)
        self.H = H
        self.N = N
        self.rw_len = rw_len
        self.H_inputs = H_inputs

    def forward(self, x):
        '''
        forward method in class Discriminator
        you already know what this does
        '''
        x = x.view(-1, self.N)
        xa = self.W_down(x)
        xa = xa.view(-1, self.rw_len, self.H_inputs)
        hc = self.init_hidden(xa.size(0))
        for i in range(self.rw_len):
            hc = self.lstmcell(xa[:, i, :], hc)
        out = hc[0]
        pred = self.lin_out(out)
        return pred

    def init_inputs(self, num_samples):
        '''
        self explanatory: initializes inputs
        '''
        weight = next(self.parameters()).data
        return weight.new(num_samples, self.H_inputs).zero_().type(torch.float64)

    def init_hidden(self, num_samples):
        '''
        self explanatory: initializes hidden layers
        '''
        weight = next(self.parameters()).data
        return (weight.new(num_samples, self.H).zero_().contiguous().type(torch.float64), weight.new(num_samples, self.H).zero_().contiguous().type(torch.float64))