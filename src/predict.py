import torch
import model
import predictor


device = torch.device('cpu')
feature_steps = 3
training_path = '../training/ENGSTR1/rnn1_ep40_t1.pt'

raw_sequence = input('Give a sentence of %d words and press Enter:\n'%(feature_steps))

training = torch.load(training_path, map_location=device)
vocab = training['vocab']
RNN = model.rnn1(n_inp=len( vocab.get_itos() ), n_steps=feature_steps, device=device)
RNN.load_state_dict(training['model_params'])

import pdb; pdb.set_trace()

predictor1 = predictor.predictor(model=RNN, vocab=vocab, device=device)
predictor1.top_and_next_word(x=raw_sequence, n_possibilities=5)