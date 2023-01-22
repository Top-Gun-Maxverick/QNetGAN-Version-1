import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import *
import pennylane as qml

#SOURCE: https://github.com/rdisipio/qlstm/blob/main/qlstm_pennylane.py and me
class QLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits, n_qlayers=1, backend="default.qubit"):
        super(QLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"

        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_input = [f"wire_input_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]

        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_input = qml.device(self.backend, wires=self.wires_input)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_output = qml.device(self.backend, wires=self.wires_output)

        def _circuit_forget(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_forget)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_forget]
        self.qlayer_forget = qml.QNode(_circuit_forget, self.dev_forget, interface="torch")

        def _circuit_input(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_input)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_input, rotation=qml.RY)
            #qml.templates.BasicEntanglerLayers(weights, wires=self.wires_input, rotation=qml.RZ)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_input]
        self.qlayer_input = qml.QNode(_circuit_input, self.dev_input, interface="torch")

        def _circuit_update(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_update)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_update)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_update]
        self.qlayer_update = qml.QNode(_circuit_update, self.dev_update, interface="torch")

        def _circuit_output(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_output)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_output)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_output]
        self.qlayer_output = qml.QNode(_circuit_output, self.dev_output, interface="torch")

        weight_shapes = {"weights": (n_qlayers, n_qubits)}
        #print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")

        self.cell = torch.nn.Linear(self.input_size+self.hidden_size, n_qubits, bias=True)
        torch.nn.init.xavier_uniform_(self.cell.weight)
        torch.nn.init.zeros_(self.cell.bias)
        
        self.VQC = {
            'forget': qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
            'input': qml.qnn.TorchLayer(self.qlayer_input, weight_shapes),
            'update': qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
            'output': qml.qnn.TorchLayer(self.qlayer_output, weight_shapes)
        }
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)

    def forward(self, x, hidden): #, init_states=None):
        hx, cx = hidden
        gates = torch.cat((x, hx), dim=1)
        gates = self.cell(gates)

        for layer in range(self.n_qlayers):
            ingate = torch.sigmoid(self.clayer_out(self.VQC['forget'](gates)))  # forget block
            forgetgate = torch.sigmoid(self.clayer_out(self.VQC['input'](gates)))  # input block
            cellgate = torch.tanh(self.clayer_out(self.VQC['update'](gates)))  # update block
            outgate = torch.sigmoid(self.clayer_out(self.VQC['output'](gates))) # output block

            cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
            hy = torch.mul(outgate, torch.tanh(cy))

        return (hy, cy)
            
        
#MOSTLY COPIED FROM NETGAN GENERATOR
class QGenerator(nn.Module):
    def __init__(self, H_inputs, H, z_dim, N, rw_len, temp):
        '''
            H_inputs: input dimension
            H:        hidden dimension
            z_dim:    latent dimension
            N:        number of nodes (needed for the up and down projection)
            rw_len:   number of LSTM cells
            temp:     temperature for the gumbel softmax
        '''
        super(QGenerator, self).__init__()
        self.intermediate = nn.Linear(z_dim, H).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.intermediate.weight)
        torch.nn.init.zeros_(self.intermediate.bias)
        self.c_up = nn.Linear(H, H).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.c_up.weight)
        torch.nn.init.zeros_(self.c_up.bias)
        self.h_up = nn.Linear(H, H).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.h_up.weight)
        torch.nn.init.zeros_(self.h_up.bias)
        self.lstmcell = QLSTMCell(input_size=H_inputs, hidden_size=H, n_qubits=N).type(torch.float64)
        self.W_up = nn.Linear(H, N).type(torch.float64)
        self.W_down = nn.Linear(N, H_inputs, bias=False).type(torch.float64)
        self.rw_len = rw_len
        self.temp = temp
        self.H = H
        self.latent_dim = z_dim
        self.N = N
        self.H_inputs = H_inputs
        
    def forward(self, latent, inputs, device='cuda'):
        intermediate = torch.tanh(self.intermediate(latent))
        hc = (torch.tanh(self.h_up(intermediate)), torch.tanh(self.c_up(intermediate)))
        out = []  
        for i in range(self.rw_len):
            hh, cc = self.lstmcell(inputs, hc)
            hc = (hh, cc)
            h_up = self.W_up(hh)                
            h_sample = self.gumbel_softmax_sample(h_up, self.temp, device)
            inputs = self.W_down(h_sample)      
            out.append(h_sample)
        return torch.stack(out, dim=1)

    def sample_latent(self, num_samples, device):
        return torch.randn((num_samples, self.latent_dim)).type(torch.float64).to(device)


    def sample(self, num_samples, device):
        noise = self.sample_latent(num_samples, device)
        input_zeros = self.init_hidden(num_samples).contiguous().type(torch.float64).to(device)
        generated_data = self(noise,  input_zeros, device)
        return generated_data

    def sample_discrete(self, num_samples, device):
        with torch.no_grad():
            proba = self.sample(num_samples, device)
        return np.argmax(proba.cpu().numpy(), axis=2)

    def sample_gumbel(self, logits, eps=1e-20):
        U = torch.rand(logits.shape, dtype=torch.float64)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits,  temperature, device, hard=True):
        gumbel = self.sample_gumbel(logits).type(torch.float64).to(device)
        y = logits + gumbel
        y = torch.nn.functional.softmax(y / temperature, dim=1)
        if hard:
            y_hard = torch.max(y, 1, keepdim=True)[0].eq(y).type(torch.float64).to(device)
            y = (y_hard - y).detach() + y
        return y

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return weight.new(batch_size, self.H_inputs).zero_().type(torch.float64)