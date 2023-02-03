import torch
from torch import nn


class embed(nn.Module):

    def __init__(self, vocab_size, n_emb, n_steps, device):

        super().__init__()

        self.device = device

        self.vocab_size = vocab_size
        self.n_emb = n_emb
        self.n_steps = n_steps
        init_std = 1/(self.vocab_size*self.n_emb)

        self.W = nn.Parameter(torch.randn(self.vocab_size, self.n_emb).to(self.device) * init_std)
        self.b = nn.Parameter(torch.zeros(self.n_emb).to(self.device))

    def forward(self, inputs):
        '''
        Inputs:
            <inputs>: Type: torch.Tensor. Shape: (n_examples, n_steps, vocab_size).

        Outputs:
            <outputs>: Type: torch.Tensor. Shape: (n_examples, n_steps, n_emb)
        '''

        outputs = []
        for (i, X_t) in enumerate(inputs.swapaxes(0,1)):
            O_t = torch.matmul(X_t, self.W) + self.b
            outputs.append(O_t)
        outputs = torch.stack(outputs).swapaxes(0,1)

        return outputs

class vanilla_recurrent(nn.Module):
    """
    Description:
        Vanilla recurrent layer.
    """

    def __init__(self, n_inp, n_hid, device):

        super().__init__()

        self.device = device

        self.n_hid = n_hid
        init_std = 1/(n_inp*n_hid)

        self.W_xh = nn.Parameter(torch.randn(n_inp, n_hid).to(self.device) * init_std)
        self.W_hh = nn.Parameter(torch.randn(n_hid, n_hid).to(self.device) * init_std)
        self.b_h = nn.Parameter(torch.zeros(n_hid).to(self.device))

    def forward(self, inputs, H_t=None):
        """
        Inputs:
            <inputs>: Type: torch.Tensor. Shape: (n_minibatch, n_steps, n_inp).
            <H_t>: Type: torch.Tensor. Shape: (n_minibatch, n_hid).

        Outputs:
            <outputs>: Type: torch.Tensor. Shape: (n_minibatch, n_steps, n_hid).
        """

        if H_t is None:
            H_t = torch.zeros((inputs.shape[0], self.n_hid)).to(self.device)
        outputs = []
        for (t, X_t) in enumerate(inputs.swapaxes(0, 1)):
            H_t = torch.tanh(torch.matmul(X_t, self.W_xh) + torch.matmul(H_t, self.W_hh) + self.b_h)
            outputs.append(H_t)
        outputs = torch.stack(outputs).swapaxes(0, 1)

        return outputs, H_t

class lstm1(nn.Module):

    def __init__(self, n_inp, n_hid, device):
        
        super().__init__()

        self.device = device
        
        self.n_inp = n_inp
        self.n_hid = n_hid
        
        init_weight = lambda *shape: nn.Parameter(torch.randn(*shape).to(self.device) * 1/(shape[0] * shape[1]))
        triple = lambda: \
        (
            init_weight(self.n_inp, self.n_hid),
            init_weight(self.n_hid, self.n_hid),
            nn.Parameter(torch.zeros(self.n_hid))
        )
        self.W_xi, self.W_hi, self.b_i = triple() # Input gate
        self.W_xf, self.W_hf, self.b_f = triple() # Forget gate
        self.W_xo, self.W_ho, self.b_o = triple() # Output gate
        self.W_xc, self.W_hc, self.b_c = triple() # Input node

    def forward(self, inputs, H_C_t=None):
        if H_C_t is None:
            # Initial state with shape: (self.batch_size, self.n_hid)
            H_t = torch.zeros((inputs.shape[0], self.n_hid), device=self.device)
            C_t = torch.zeros((inputs.shape[0], self.n_hid), device=self.device)
        else:
            H_t, C_t = H_C_t
        outputs = []
        for (t, X_t) in enumerate(inputs.swapaxes(0, 1)):
            I_t = torch.sigmoid(torch.matmul(X_t, self.W_xi) + torch.matmul(H_t, self.W_hi) + self.b_i)
            F_t = torch.sigmoid(torch.matmul(X_t, self.W_xf) + torch.matmul(H_t, self.W_hf) + self.b_f)
            O_t = torch.sigmoid(torch.matmul(X_t, self.W_xo) + torch.matmul(H_t, self.W_ho) + self.b_o)
            C_tilde_t = torch.tanh(torch.matmul(X_t, self.W_xc) + torch.matmul(H_t, self.W_hc) + self.b_c)
            C_t = F_t * C_t + I_t * C_tilde_t
            H_t = O_t * torch.tanh(C_t)
            outputs.append(H_t)
        outputs = torch.stack(outputs).swapaxes(0, 1)

        return outputs, (H_t, C_t)

class gru1(nn.Module):

    def __init__(self, n_inp, n_hid, device, dropout_rate=0.0):
        
        super().__init__()

        self.device = device
        
        self.n_inp = n_inp
        self.n_hid = n_hid
        
        self.dropout_rate = dropout_rate
        
        init_weight = lambda *shape: nn.Parameter(torch.randn(*shape).to(self.device) * 1/(shape[0] * shape[1]))
        triple = lambda: \
        (
            init_weight(self.n_inp, self.n_hid),
            init_weight(self.n_hid, self.n_hid),
            nn.Parameter(torch.zeros(self.n_hid).to(self.device))
        )
        self.W_xz, self.W_hz, self.b_z = triple() # Update gate
        self.W_xr, self.W_hr, self.b_r = triple() # Reset gate
        self.W_xh, self.W_hh, self.b_h = triple() # Candidate hidden state

    def forward(self, inputs, H_t=None, train=False):

        if H_t is None:
            # Initial state with shape: (self.batch_size, self.n_hid)
            H_t = torch.zeros((inputs.shape[0], self.n_hid), device=self.device)
        outputs = []
        for (t, X_t) in enumerate(inputs.swapaxes(0, 1)):
            Z_t = torch.sigmoid(torch.matmul(X_t, self.W_xz) + torch.matmul(H_t, self.W_hz) + self.b_z)
            R_t = torch.sigmoid(torch.matmul(X_t, self.W_xr) + torch.matmul(H_t, self.W_hr) + self.b_r)
            H_tilde_t = torch.tanh(torch.matmul(X_t, self.W_xh) + torch.matmul(R_t * H_t, self.W_hh) + self.b_h)
            H_t = Z_t * H_t + (1 - Z_t) * H_tilde_t
            outputs.append(H_t)
        outputs = torch.stack(outputs).swapaxes(0, 1)

        if train and (self.dropout_rate != 0.0):

            switches = torch.ones(self.n_hid).to(self.device)
            switches[0:round(self.dropout_rate * self.n_hid)] = 0.0
            switches = switches[torch.randperm(self.n_hid)]

            switches_outputs = switches.repeat(outputs.shape[0], outputs.shape[1], 1)
            outputs = outputs * switches_outputs

            switches_H_t = switches.repeat(H_t.shape[0], 1)
            H_t = H_t * switches_H_t

        return outputs, H_t

class linear1(nn.Module):

    def __init__(self, n_inp, n_hid, device, dropout_rate=0.0):

        super().__init__()

        self.device = device

        self.n_inp = n_inp
        self.n_hid = n_hid
        init_std = 1/(self.n_inp*self.n_hid)

        self.dropout_rate = dropout_rate

        self.W = nn.Parameter(torch.randn(self.n_inp, self.n_hid).to(self.device) * init_std)
        self.b = nn.Parameter(torch.zeros(self.n_hid).to(self.device))

    def forward(self, inputs, train=False):

        outputs = torch.matmul(inputs, self.W) + self.b
        if train and (self.dropout_rate != 0.0):

            switches = torch.ones(self.n_hid).to(self.device)
            switches[0:round(self.dropout_rate * self.n_hid)] = 0.0
            switches = switches[torch.randperm(self.n_hid)]

            switches_outputs = switches.repeat(outputs.shape[0], 1)
            outputs = outputs * switches_outputs

        return outputs