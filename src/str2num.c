#include <stdio.h>
#include <string.h>


void str_mod(char **seq, int seq_length, char **vocab_itos, int vocab_itos_length, int *enc_seq)
{
    int word_exists, seq_word_idx, vocab_word_idx;

    for (seq_word_idx = 0; seq_word_idx < seq_length; ++seq_word_idx)
    {
        word_exists = 0;
        for (vocab_word_idx = 0; vocab_word_idx < vocab_itos_length; ++vocab_word_idx)
        {
            if (strcmp(*(seq+seq_word_idx), *(vocab_itos+vocab_word_idx)) == 0)
            {
                *(enc_seq+seq_word_idx) = vocab_word_idx;
                word_exists = 1;
                break;
            }
        }
        if (!word_exists)
        {
            *(enc_seq+seq_word_idx) = 0;
        }
    }
}