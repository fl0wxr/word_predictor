from torch import nn
import layer


class rnn1(nn.Module):

    def __init__(self, n_inp, n_steps, device):

        super().__init__()

        self.device = device

        self.n_inp = n_inp
        self.n_emb = 100
        self.n_hid1 = 500
        self.n_steps = n_steps

        self.emb = layer.embed(self.n_inp, self.n_emb, n_steps=self.n_steps, device=self.device)
        self.rec1 = layer.gru1(self.n_emb, self.n_hid1, device=self.device, dropout_rate=0.0)
        self.out_linear = layer.linear1(self.n_hid1, self.n_inp, device=self.device, dropout_rate=0.0)
        self.softmax = nn.Softmax(dim=1)

        self.name = 'rnn1'

    def forward(self, X, train=False):

        out = self.emb(X)
        out = self.rec1(out, train=train)
        out = self.out_linear(out[1], train=train)
        out = self.softmax(out)

        return out