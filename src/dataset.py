import torch
import torchtext
import re
from collections import Counter, OrderedDict
import ctypes
import requests
import plot


def c_numericalize_str(seq, vocab_itos):
    '''
    Description:
        Converts a sequence of string-tokens to a sequence of vocabulary-index-tokens, while maintaining their order. It replicates the sequence, whilst replacing the string tokens with index tokens with respect to a vocabulary.
        - Advantages:
        1. ~287 times faster than <p_numericalize_str> .
        2. Verified to work with UTF-8 symbols.

    Inputs:
        <seq>: Type: <list[<str>]>. The sequence of words to be transformed/numericalized.
        <vocab_itos>: Type: <list[<str>]>. The list containing all the vocabularies string-tokens at the position with index equal to vocabulary-index-token. Must include the <unk> character in the beginning.

    Returns:
        <enc_seq>: Type: <list[<int>]>. Contains <seq> with each string replaced with the vocabularies index.
    '''

    ## Load the shared library
    lib = ctypes.CDLL('../lib/str2num.so')

    ## Define the prototype
    lib.str_mod.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]

    enc_seq = [0 for i in range(len(seq))]

    seq_array = (ctypes.c_char_p * len(seq))()
    enc_seq_array = (ctypes.c_int * len(enc_seq))()
    vocab_itos_array = (ctypes.c_char_p * len(vocab_itos))()

    for i, s in enumerate(seq):
        seq_array[i] = s.encode()

    for i, s in enumerate(vocab_itos):
        vocab_itos_array[i] = s.encode()

    lib.str_mod(seq_array, len(seq_array), vocab_itos_array, len(vocab_itos_array), enc_seq_array)

    enc_seq = list(enc_seq_array)

    return enc_seq

def p_numericalize_str(seq, vocab_):

    print('Conversion Status:')
    seq_int = []
    div = 1000
    for idx, word in enumerate(seq):
        if word not in vocab_.get_itos():
            seq_int.append(vocab_.get_stoi()['<unk>'])
        else:
            seq_int.append(vocab_.get_stoi()[word])

        if (idx+1) % div == 0:
            print('Progress: %.2f %%'%(100*(idx+1) / len(seq)))
    print('Progress: 100 %')
    print('Conversion successful.')

    return seq_int

def tokenize(raw_dataset_str):

    ## Raw text preprocessing
    dataset_str_prep = re.sub('[^A-Za-z.]+', ' ', raw_dataset_str).lower()
    dataset_str_prep = re.sub('[.]+', ' . ', dataset_str_prep)
    dataset_str_prep = re.sub('[ ]+', ' ', dataset_str_prep)
    dataset_seq_str = dataset_str_prep.split(' ')
    if dataset_seq_str[0] == '':
        dataset_seq_str = dataset_seq_str[1:]
    if dataset_seq_str[-1] == '':
        dataset_seq_str = dataset_seq_str[:-1]

    return dataset_seq_str

def build_vocab(seq_string):
    """
    Description:
        Generates a vocabulary object with respect to the decreasing order of token-frequency.
    
    Inputs:
        <seq_string>: Type: list. Contains the list containing the dataset's tokens.

    Outputs:
        <vocab_>: Type: <torchtext.vocab.Vocab>.
        <freqs>: Type: <list>.
    """

    seq_string_counter = Counter(seq_string)
    seq_string_sorted_by_freq_tuples = sorted(seq_string_counter.items(), key=lambda x: x[1], reverse=True)
    seq_string_ordered_dict = OrderedDict(seq_string_sorted_by_freq_tuples)
    vocab_ = torchtext.vocab.vocab\
    (
        ordered_dict = seq_string_ordered_dict,
        min_freq = 3,
        specials = ['<unk>'],
        special_first = True
    )

    freqs = list(iter(seq_string_ordered_dict.values()))

    return freqs, vocab_

def convert_strings(dataset_seq_str_, vocab_):
    """
    Description:
        Converts string tokens from a subscriptable sequence of strings to their vocabularies corresponding integers and appends these values to a sequence. Takes into account words that exist in the data sequence but are absent from the vocabulary.

    Inputs:
        <dataset_seq_str_>: Type: <list[<str>]>. Contains the string tokens.
        <vocab_>: Type: <torchtext.vocab.Vocab>. The vocabulary.
    """

    dataset_seq_int = c_numericalize_str(dataset_seq_str_, vocab_.get_itos())

    return dataset_seq_int


class text_dataset:
    """
    Description:
        Tokens are defined to be words.
    """

    def __init__(self):

        def parse_local_raw_data(raw_dataset_path):
        
            with open(raw_dataset_path, 'r') as file:
                raw_dataset_str = file.read()

            return raw_dataset_str

        def parse_web_raw_data(url):

            response = requests.get(url)
            raw_dataset_str = response.text

            return raw_dataset_str

        raw_dataset_path = '../datasets/ENGSTR1.txt'
        self.dataset_name = raw_dataset_path.split('.')[-2].split('/')[-1]

        ## [training fraction, validation fraction, test fraction].
        self.split_fractions = [0.9, 0.1, 0.0]

        self.raw_dataset_str = parse_local_raw_data(raw_dataset_path)#[:10000]

    def partition(self, feature_steps, prediction_steps):
        """
        Description:
            Used to build <X> and <y> from dataset for the problem of predicting words after a sequence of feature-words.

        Input:
            <feature_steps>: Type: <int>.
            <prediction_steps>: Type: <int>.

        Output:
            <X>: Type: <torch.Tensor>. Shape: (number of examples, <feature_steps>).
            <y>: Type: <torch.Tensor>. Shape: (number of examples, <prediction_steps>) or (number of examples) if <prediction_steps> equals to 1.
        """

        X = []
        y = []
        for idx in range( len(self.dataset_seq_int) ):
            if (idx+feature_steps)+prediction_steps == len(self.dataset_seq_int):
                break
            X.append(self.dataset_seq_int[idx:idx+feature_steps])
            y.append(self.dataset_seq_int[idx+feature_steps:(idx+feature_steps)+prediction_steps])

        X = torch.stack(X)
        y = torch.stack(y)
        if y.shape[1] == 1:
            y = y[:,0]

        return X, y

    def generate_dataset(self, feature_steps, prediction_steps):
        """
        Description:
            Loads, conducts basic token preprocessing and generates the feature and target tensors.
        """

        self.dataset_seq_str = tokenize(self.raw_dataset_str)
        ## Vocabulary construction
        freqs, self.vocab = build_vocab(seq_string = self.dataset_seq_str)
        # plot.plot_frequency_curve(freqs, self.dataset_name)

        self.dataset_seq_int = torch.tensor(convert_strings(self.dataset_seq_str, self.vocab))

        self.feature_steps = feature_steps
        self.prediction_steps = prediction_steps
        self.vocab_size = len( self.vocab.get_itos() )

        X, y = self.partition(feature_steps, prediction_steps)
        
        ## Shuffle instance positions
        perm = torch.randperm(X.shape[0])
        X = X[perm,...]
        y = y[perm,...]

        self.n_instances = y.shape[0]

        self.dataset = torch.utils.data.TensorDataset(X, y)
        if self.split_fractions[-1] != 0: # Included a test set
            self.train_set, self.val_set, self.test_set = torch.utils.data.random_split(self.dataset, self.split_fractions, generator=torch.Generator().manual_seed(42))

            self.n_train = len(self.train_set)
            self.n_val = len(self.val_set)
            self.n_test = len(self.test_set)

            separateXy = lambda set: ( torch.stack([set[i][_] for i in range(len(set))], axis=0) for _ in range(2) )

            self.X_train, self.y_train = separateXy(self.train_set)
            self.X_val, self.y_val = separateXy(self.val_set)
            self.X_test, self.y_test = separateXy(self.test_set)

        else: # Excluded a test set
            self.train_set, self.val_set = torch.utils.data.random_split(self.dataset, self.split_fractions[:2], generator=torch.Generator().manual_seed(42))

            self.n_train = len(self.train_set)
            self.n_val = len(self.val_set)

            separateXy = lambda set: ( torch.stack([set[i][_] for i in range(len(set))], axis=0) for _ in range(2) )

            self.X_train, self.y_train = separateXy(self.train_set)
            self.X_val, self.y_val = separateXy(self.val_set)