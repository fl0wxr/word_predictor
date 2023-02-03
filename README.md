# word_predictor

## Introduction

word_predictor is a PyTorch-based training program for Linux systems that trains word predictors. Such word predictors accept as input a sequence of `T` words, and predicts a probability distribution with respect to a vocabulary sample space. The vocabulary is created from the same corpus that is used to create the training and validation sets. When given a sequence of - say 3 words, the trained model assigns a probability to each word in the vocabulary to determine the most appropriate fourth word to follow. The design of the recurrent layers (found in `layer.py`) was heavily influenced by [[1][d2l]].

## Standard Corpus Preprocessing

Non-alphabet and non-period characters are removed from the corpus prior to tokenization. Further, the corpus is segmented into tokens from which the vocabulary is constructed. Tokens that appear only once in the corpus are replaced by `'<unk>'`. This special token is used to replace all input words that are not part of the vocabulary. Each string-token is mapped into a unique integer. Seeing the vocabulary as a vector, these integers are indices of that vector, and the words are sorted in a decreasing order with respect to the frequency of appearance inside the corpus. The more frequent a token is, the lower its vocabulary-index will be. The string-to-index mapping function is implemented using ctypes to improve performance. To build the feature and target tensors, imagine a window of size $T+1$ scanning the segmented corpus from start to finish. An instance $i \in \{ 1, \dots, n \}$ represents one such unique scan. The first $T$ words $x_1^i,...,x_T^i$ are concatenated together to form the feature vector $\mathbf{x}^i$. While on the other hand, the word at position $T+1$ is taken as the ground truth, or $y^i$ in this case.

One hot encoding is performed minibatch-wise during training to prevent potential memory overflows. For example, if the vocabulary contains 3,000 words and the dataset has 300,000 instances with 3 features and 1 target token, storing the data as one hot encodings (OHEs) would require 14.4 GB of memory, which exceeds the VRAM capacity of many commercial GPUs. To resolve this, the dataset is stored in the previously described integer format, and OHEs are only invoked within the training loop for a single minibatch.

## Shallow Architecture Effectiveness

As the trainings of shallow, more linear architectures consistently led to an overfitting behaviour, it was assumed that deeper and more complex ones will inevitably magnify that negative effect. Hence, the layer-wise depth was limited to match that of `rnn1`.

`rnn1` is the token-predictor architecture, which was used to train the currently provided demo model `rnn1_ep40_t1`, located at `./training/ENGSTR1/rnn1_ep40_t1.pt`. Its embedding layer, allows the neural network to harvest each token's relation with the neighboring words. After the embedding layer, a GRU layer is used to incorporate temporal information and effectively construct context, similar to the way a human would when reading a sentence word by word. It's worth noting that benefits of recurrent layers are more pronounced with increased time steps in the feature-inputs. Finally a fully connected layer stacked with a softmax function generates the associated probability distribution.

## Checkpoints

During training, the models and information about their training are periodically saved inside `./training`. Every time that this happens a `.pt` and a `.png` file are stored. The `.png` demonstrates the effectiveness of the model throughout past epochs. Additionally, the `.pt` file, when parsed through torch's load method the resulting variable is a dictionary that containing the following items:

```
{
    'model_params': <`model.state_dict()` holding the model's architecture and trainable parameters>,
    'vocab': <The vocabulary as `torchtext.vocab`>,
    'data_info':
    {
        'n_train': <Number of training instances>,
        'n_val': <Number of validation instances>
    },
    'metrics_history':
    {
        'loss':
        {
            'train': <`list` containing the training loss history>,
            'val': <`list` containing the validation loss history>
        },
        'accuracy':
        {
            'train': <`list` containing the training accuracy history>,
            'val': <`list` containing the validation accuracy history>
        }
    },
    'training_hparams':
    {
        'epoch': <Training algorithm's latest epoch>,
        'learning_rate': <Training algorithm's learning rate>,
        'minibatch_size': <Training algorithm's minibatch size>
    },
    'delta_t': <Time required to train the model>
}
```

The reason that `model.state_dict()` was prefered to be stored in the `.pt` file instead of `model` itself, was to allow model information to be parsed in different machines without worrying about deserialization errors due to CUDA or PyTorch version incompatibilies.

## ENGSTR1 Dataset

For effective training of word predictors, it is crucial to have a large dataset that captures diverse patterns relevant to the problem at hand. As an attempt to achieve this, it was decided to combine multiple corpuses sourced from [[2][gutenberg]]. The resulting corpus was named `ENGSTR1`, which includes:

1. The Time Machine, by H. G. Wells <sup>[[b1][book1]]</sup>

2. Middlemarch, by George Eliot <sup>[[b2][book2]]</sup>

3. Pride and prejudice, by Jane Austen <sup>[[b3][book3]]</sup>

4. Romeo and Juliet, by William Shakespeare <sup>[[b4][book4]]</sup>

5. A Room With A View, by E. M. Forster <sup>[[b5][book5]]</sup>

6. The Enchanted April, by Elizabeth Von Arnim <sup>[[b6][book6]]</sup>

The resulting preprocessed dataset is split to 835,460 training instances (90% of the dataset) and 92,828 validation instances (10% of the dataset). Each feature vector input has 3 token-steps in total. It should be noted that the dataset is imbalanced as it can be observed in the following frequency plot.

<div align="center">
    <img width="60%" src="https://raw.githubusercontent.com/fl0wxr/word_predictor/master/datasets/ENGSTR1.png">
</div>

## Demo rnn1_ep40_t1

Predictions of `rnn1_ep40_t1` are produced with an input sentence composed of 3 words. This model was trained on `ENGSTR1` with a minibatch size set to 2<sup>6</sup> or 64, in 40 epochs. The vocabulary size is 12217. Adam optimized the model with respect to the Categorical Cross Entropy loss function. The neural network's architecture configuration can be found at `layer.py`, at the `rnn1` class. By running `predict.py` with `training_path = './training/ENGSTR1/rnn1_ep40_t1.pt'`

```
Give a sentence of 3 words and press Enter:
```

the program asks us to type in a sentence. The sentence has to have no more than 3 words (the period counts as a word too!). By entering

```
Carefully placed at
```

it prints

```
Given sentence: Carefully placed at
Predicted word: her
Complete sentence: Carefully placed at her
Size of vocabulary: 12217
Top 5 predicted words:
1. 'her': 58.696222 %
2. 'the': 23.382269 %
3. 'his': 6.458900 %
4. 'once': 4.324654 %
5. 'them': 1.002752 %
```

This shows the 5 most likely continuations suggested by `rnn1_ep40_t1`.

<div align="center">
    <img width="100%" src="https://raw.githubusercontent.com/fl0wxr/word_predictor/master/training/ENGSTR1/rnn1_ep40_t1.png">
</div>

The accuracy of the model on the validation set is ~13.8%, which is worse than the maximum ~17.7% at epoch 4. Αfter that epoch, overfitting makes its presence noticeable. It took approximately 2 hours and 41 minutes to complete the training.

## Usage

In order to use this software first you should navigate in `./src`. From there type and enter
```
gcc str2num.c -o ../lib/str2num.so -shared
```
The main training program is `train.py`. You can execute it by running
```
python3 ./train.py
```
and to predict use `predict.py` through
```
python3 ./predict.py
```

## Software Versions and Hardware

The training was carried out on a Google Colab<sup>[[3][google_colab]]</sup> machine using a Tesla T4 GPU - 16 GB VRAM, Intel(R) Xeon(R) CPU (2.20GHz) - 12.7 GB RAM.  The following software versions were used:

- Python v3.8.10
- CUDA v11.6
- torch v1.13.1+cu116

## Conclusion and Improvement Ideas

Model performance could be improved by increasing the number of steps. This would provide additional information about the dataset, and that is because the dataset consists of longer stories. As a result, there are a lot of exploitable long term dependencies, contrasting the contextual simplicity of multiple short dialogues between two people about the same subject. The current limited context provided by three words is insufficient to help the model understand the dataset's stories. That's potentially how the overfitting is justified. Regularization techniques like dropout, in this case, does more harm than good, as the training process does not have enough context to effectively learn from the dataset.

[d2l]: <https://d2l.ai/d2l-en.pdf>

[gutenberg]: <https://www.gutenberg.org/browse/scores/top>

[book1]: <https://www.gutenberg.org/ebooks/35>

[book2]: <https://www.gutenberg.org/ebooks/145>

[book3]: <https://www.gutenberg.org/ebooks/1342>

[book4]: <https://www.gutenberg.org/ebooks/1513>

[book5]: <https://www.gutenberg.org/ebooks/2641>

[book6]: <https://www.gutenberg.org/ebooks/16389>

[google_colab]: <https://colab.research.google.com/>