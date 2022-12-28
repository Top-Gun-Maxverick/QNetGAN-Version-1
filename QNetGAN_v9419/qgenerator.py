import torch
import torch.nn as nn
import pennylane as qml

#SOURCE: https://github.com/rdisipio/qlstm/blob/main/qlstm_pennylane.py
class QLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits=4, n_qlayers=1, batch_first=True, return_sequences=False, return_state=False, backend="default.qubit"):
        super(QLSTM, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend 
        print("Other options for self.backend:", "qiskit.basicaer", "qiskit.ibm(q)")

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        #self.dev = qml.device("default.qubit", wires=self.n_qubits)
        #self.dev = qml.device('qiskit.basicaer', wires=self.n_qubits)
        #self.dev = qml.device('qiskit.ibm', wires=self.n_qubits)
        # use 'qiskit.ibmq' instead to run on hardware

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
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_input)
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
        print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")

        self.clayer_in = torch.nn.Linear(self.concat_size, n_qubits)
        self.VQC = {
            'forget': qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
            'input': qml.qnn.TorchLayer(self.qlayer_input, weight_shapes),
            'update': qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
            'output': qml.qnn.TorchLayer(self.qlayer_output, weight_shapes)
        }
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)
        #self.clayer_out = [torch.nn.Linear(n_qubits, self.hidden_size) for _ in range(4)]

    def forward(self, x, init_states=None):
        '''
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        '''
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size)  # hidden state (output)
            c_t = torch.zeros(batch_size, self.hidden_size)  # cell state
        else:
            # for now we ignore the fact that in PyTorch you can stack multiple RNNs
            # so we take only the first elements of the init_states tuple init_states[0][0], init_states[1][0]
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]
            
            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)

            # match qubit dimension
            y_t = self.clayer_in(v_t)

            f_t = torch.sigmoid(self.clayer_out(self.VQC['forget'](y_t)))  # forget block
            i_t = torch.sigmoid(self.clayer_out(self.VQC['input'](y_t)))  # input block
            g_t = torch.tanh(self.clayer_out(self.VQC['update'](y_t)))  # update block
            o_t = torch.sigmoid(self.clayer_out(self.VQC['output'](y_t))) # output block

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
            
        
#THIS CODE IS SUPER SCUFFED IDK IF IT WORKS
#MOSTLY COPIED FROM NETGAN GENERATOR
class QGenerator(nn.Module):
    def __init__(self, H_inputs, H, z_dim, N, rw_len, temp):
        '''
            H_inputs: input dimension
            H:        hidden dimension
            z_dim:    latent dimension
            N:        number of nodes/qubits (needed for the up and down projection)
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
        self.lstmcell = QLSTM(H_inputs, H, n_qubits = N).type(torch.float64)
        self.W_up = nn.Linear(H, N).type(torch.float64)
        self.W_down = nn.Linear(N, H_inputs, bias=False).type(torch.float64)
        self.rw_len = rw_len
        self.temp = temp
        self.H = H
        self.latent_dim = z_dim
        self.N = N
        self.H_inputs = H_inputs
        
    def forward(self, latent, inputs, device='cuda', backend='default.qubit'):
        intermediate = torch.tanh(self.intermediate(latent))
        hc = (torch.tanh(self.h_up(intermediate)), torch.tanh(self.c_up(intermediate)))
        out = []  # gumbel_noise = uniform noise [0, 1]
        for i in range(self.rw_len):
            hidden_seq, hc = self.lstmcell(inputs, hc)
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
        """ Draw a sample from the Gumbel-Softmax distribution"""
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