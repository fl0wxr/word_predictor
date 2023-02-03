import torch
from torch.nn import functional as F
import dataset


class predictor:

    def __init__(self, model, vocab, device):
        self.model = model
        self.device = device
        self.tokenizer = dataset.tokenize
        self.numericalize = dataset.c_numericalize_str
        self.vocab = vocab
        self.vocab_size = len( self.vocab.get_itos() )
        self.ohe = lambda x: torch.stack([F.one_hot(x[seq_idx], self.vocab_size) for seq_idx in range(len(x))], axis=0).type(torch.float32)

    def preprocess(self, x):

        x = self.tokenizer(x)
        x = torch.Tensor(dataset.c_numericalize_str(x, self.vocab.get_itos())).type(torch.int64).to(self.device)
        x = self.ohe(x)[None,...]

        return x

    def next_word(self, x):

        x_ = self.preprocess(x)

        with torch.no_grad():
            p_distr = self.model(x_)
        predicted_word_int = torch.argmax(p_distr, axis=1)
        predicted_word_str = self.vocab.get_itos()[predicted_word_int]
        print('Given sentence: ' + x)
        print('Predicted word: ' + predicted_word_str)
        print('Complete sentence: ' + x + ' ' + predicted_word_str)

    def top(self, x, n_possibilities):

        x_ = self.preprocess(x)
        with torch.no_grad():
            p_distr = self.model(x_)
        predicted_words_int = torch.topk(input=p_distr, k=n_possibilities)
        predicted_words_str = [self.vocab.get_itos()[predicted_words_int[1][0][possibility]] for possibility in range(n_possibilities)]
        p_distr_predicted_words = predicted_words_int[0][0].tolist()

        top_words = {predicted_words_str[possibility]: p_distr_predicted_words[possibility] for possibility in range(n_possibilities)}

        print('Size of vocabulary: %d'%(self.vocab_size))
        print('Top %d predicted words:'%n_possibilities)
        for (possibility, (word, model_probability)) in enumerate(top_words.items()):
            print('%d. \'%s\': %f %%'%(possibility+1, word, model_probability*100))

    def top_and_next_word(self, x, n_possibilities):
        self.next_word(x)
        self.top(x, n_possibilities)