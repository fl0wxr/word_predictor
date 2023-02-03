import torch
import dataset
import trainer
import model


def load_and_train_model(training_path, vocab_size, feature_steps, data, device):
    training = torch.load(training_path, map_location=device)
    RNN = model.rnn1(n_inp=vocab_size, n_steps=feature_steps, device=device)
    RNN.load_state_dict(training['model_params'])
    trainer0 = trainer.rnn_trainer0(RNN, data, device=device)
    loss_history = \
    [
        training['metrics_history']['loss']['train'], training['metrics_history']['loss']['val']
    ]
    accuracy_history = \
    [
        training['metrics_history']['accuracy']['train'], training['metrics_history']['accuracy']['val']
    ]
    trainer0.train(loss_history=loss_history, accuracy_history=accuracy_history)

def train_model_from_scratch(vocab_size, feature_steps, data, device):
    RNN = model.rnn1(vocab_size, n_steps=feature_steps, device=device)
    trainer0 = trainer.rnn_trainer0(RNN, data, device=device)
    trainer0.train()


device = torch.device('cuda')

feature_steps = 3
prediction_steps = 1

data = dataset.text_dataset()
data.generate_dataset(feature_steps, prediction_steps)
train_model_from_scratch(data.vocab_size, feature_steps, data, device)
# load_and_train_model('./training/ENGSTR1/rnn1_ep40_t1.pt', data.vocab_size, feature_steps, data, device)