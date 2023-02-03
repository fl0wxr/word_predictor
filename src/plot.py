import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math
import numpy as np


class figure:

    def __init__(self):

        asp_ratio = 0.4375
        l = 1400
        h = asp_ratio * l
        self.fig = plt.figure(figsize=(l/96, h/96), dpi=96)

        self.ax = [None for i in range(2)]
        self.ax[0] = self.fig.add_subplot(121)
        self.ax[1] = self.fig.add_subplot(122)

    def plot(self, hor_seq, ver):

        def plot_(hor_seq, ax_idx, ver_name, ver_seq1, ver_seq2):

            ## Clearing previous frame
            self.ax[ax_idx].clear()

            ## To show only integer numbers on the x-axis
            self.ax[ax_idx].xaxis.set_major_locator(MaxNLocator(integer=True))

            self.ax[ax_idx].set_xlabel('epoch')
            self.ax[ax_idx].set_ylabel(ver_name)

            self.ax[ax_idx].plot(hor_seq, ver_seq1, color='red', label='etr %s'%(ver_name))
            self.ax[ax_idx].plot(hor_seq, ver_seq2, color='blue', label='val %s'%(ver_name))

            self.ax[ax_idx].legend()

        for (ax_idx, (ver_name, ver_seq1, ver_seq2)) in enumerate(ver):
            plot_(hor_seq, ax_idx, ver_name, ver_seq1, ver_seq2)

        ## ! Display: Begin

        # plt.pause(0.0000001)
        # self.fig.canvas.draw()

        ## ! Display: End

def plot_frequency_curve(freqs, image_name):

    word_keys = list(range(len(freqs)))
    plt.plot(word_keys, freqs, color='orange')
    # plt.title('Frequency Graph')
    plt.xlabel('Token Index')
    plt.ylabel('Frequency')
    plt.grid(visible=True, color='gray', alpha=0.5)
    plt.semilogx()
    plt.semilogy()

    plt.savefig('../datasets/'+image_name)

def plot_frequency_histogram(freqs, n_groups = 40):
    '''
    Description:
        Segments frequency list and shows frequency histogram.

    Input:
        <freqs>: Type: <list>. Contains the frequency of each one of its indices.
        <n_groups>: Type: <int>. The number of grouped indices.
    '''

    word_keys = list(range(len(freqs)))
    n_words = len(word_keys)
    group_intervals = [(math.ceil(i*n_words/n_groups), math.floor((i+1)*n_words/n_groups)) for i in range(n_groups)]
    grouped_word_freqs = [round(np.mean(freqs[group_intervals[i][0]:group_intervals[i][1]])) for i in range(n_groups)]
    grouped_word_keys = list(range(n_groups))

    plt.figure(figsize = (10, 5))
    plt.bar(grouped_word_keys, grouped_word_freqs, color='maroon', width=1, log=True, ec="k")
    plt.title('Frequency Histogram')
    plt.xlabel('Word Group')
    plt.ylabel('Frequency')
    for i, freq in enumerate(grouped_word_freqs):
        plt.text(x=i, y=freq, s=str(freq), horizontalalignment='left', rotation='horizontal')
    plt.show()