import torch
from torch.nn import functional as F
from copy import deepcopy
import numpy as np
import os
from time import time
import datetime
import plot
import metrics


class rnn_trainer0:
    """
    Description:
        Specifies the training algorithm.
    """

    def __init__(self, model_, data, device):
        """
        Inputs:
            <model_>: Type: <torch.nn.module>. Specifies the trained model's architecture and parameters.
            <data>: Type: <class>. Contains all the necessary dataset's information.
            <device>: Type: torch.device.
        """

        self.device = device

        self.epochs = 10000
        self.lr = 0.001
        self.minibatch_size = 2**6
        self.data = data
        self.model = model_

        self.bkp_freq = 1
        self.scheduled_checkpoints = {10, 50, 200, 600, 1000, 5000}

        self.training_dir_path, self.training_format = '../training/', '.pt'
        self.criterion = metrics.categ_cross_entropy
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

        self.train_dataloader = torch.utils.data.DataLoader(self.data.train_set, batch_size=self.minibatch_size, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(self.data.val_set, batch_size=2**10, shuffle=False)

    def train(self, loss_history=[[],[]], accuracy_history=[[],[]]):
        """
        Description:
            Trains a model.

        Inputs:
            <*_history>: Type: <list[list,list]>. In case the model is pretrained, this retains past history per * metric. <*_history[0]> holds the training past, and <*_history[1]> the validation past.
        """

        print('Vocabulary size: %d'%(self.data.vocab_size))
        print('Number of training instances: %d'%(self.data.n_train))

        print('Training Status:')

        val_loss_prev = float('inf')
        figure_ = plot.figure()
        initial_epoch = len(loss_history[0])
        t_before_training = time()
        for epoch in range(initial_epoch, self.epochs):
            t_i = time()
            train_minibatch_losses = []
            train_minibatch_accuracy = []
            val_minibatch_losses = []
            val_minibatch_accuracy = []
            for (i, (X_train_minibatch_i, y_train_minibatch_i)) in enumerate(self.train_dataloader):

                X_train_minibatch_i = torch.stack([F.one_hot(X_train_minibatch_i[:,seq_idx], self.data.vocab_size) for seq_idx in range(X_train_minibatch_i.shape[-1])], axis=0).swapaxes(0, 1).type(torch.float32).to(self.device)
                y_train_minibatch_i = F.one_hot(y_train_minibatch_i, self.data.vocab_size).type(torch.float32).to(self.device)
                self.optimizer.zero_grad()
                train_prediction_minibatch_i = self.model(X_train_minibatch_i, train=True)
                train_minibatch_loss = self.criterion(train_prediction_minibatch_i, y_train_minibatch_i)
                train_minibatch_losses.append(train_minibatch_loss.detach())
                train_minibatch_accuracy.append(metrics.accuracy(train_prediction_minibatch_i, y_train_minibatch_i))
                # self.clip_gradients(grad_clip_val=1, model=self.model)
                train_minibatch_loss.backward()
                self.optimizer.step()

            for (j, (X_val_minibatch_j, y_val_minibatch_j)) in enumerate(self.val_dataloader):

                X_val_minibatch_j = torch.stack([F.one_hot(X_val_minibatch_j[:,seq_idx], self.data.vocab_size) for seq_idx in range(X_val_minibatch_j.shape[-1])], axis=0).swapaxes(0, 1).type(torch.float32).to(self.device)
                y_val_minibatch_j = F.one_hot(y_val_minibatch_j, self.data.vocab_size).type(torch.float32).to(self.device)
                val_prediction_minibatch_j = self.model(X_val_minibatch_j, train=False)
                val_minibatch_losses.append(self.criterion(val_prediction_minibatch_j, y_val_minibatch_j).detach())
                val_minibatch_accuracy.append(metrics.accuracy(val_prediction_minibatch_j, y_val_minibatch_j))

            train_loss_hat = torch.sum(torch.stack(train_minibatch_losses)) / len(train_minibatch_losses)
            train_accuracy_hat = torch.sum(torch.stack(train_minibatch_accuracy)) / len(train_minibatch_accuracy)

            val_loss = torch.sum(torch.stack(val_minibatch_losses)) / len(val_minibatch_losses)
            val_accuracy = torch.sum(torch.stack(val_minibatch_accuracy)) / len(val_minibatch_accuracy)

            loss_history[0].append(train_loss_hat.item())
            loss_history[1].append(val_loss.item())

            accuracy_history[0].append(train_accuracy_hat.item())
            accuracy_history[1].append(val_accuracy.item())

            figure_.plot\
            (
                np.arange(0, epoch+1),
                (
                    ('loss', loss_history[0], loss_history[1]),
                    ('acc', accuracy_history[0], accuracy_history[1])
                )
            )

            self.save_training(epoch, metrics=(loss_history, accuracy_history), thparams=(self.lr, self.minibatch_size), t_before_training=t_before_training, figure_=figure_)

            # figure_.fig.savefig('../current_metrics.png')

            t_f = time()
            delta_t = round(t_f-t_i)
            est_next_epoch_time = datetime.datetime.utcfromtimestamp(t_f) + datetime.timedelta(seconds=delta_t)

            print('[Epoch %d @ %s]'%(epoch, datetime.datetime.utcfromtimestamp(t_f).strftime("%H:%M:%S")))
            print('Train loss: %f | Val loss: %f | Train acc: %f | Val acc: %f'%(train_loss_hat, val_loss, train_accuracy_hat, val_accuracy))
            print('Δt: %ds | Δ Val loss: %f\nNext epoch @ ~%s'%(delta_t, val_loss-val_loss_prev, est_next_epoch_time.strftime("%H:%M:%S")))

            val_loss_prev = deepcopy(val_loss)

            torch.cuda.empty_cache()

        print('Training completed.')

    def clip_gradients(self, grad_clip_val, model):
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm

    def save_training(self, epoch, metrics, thparams, t_before_training, figure_):

        def get_training_information(epoch, metrics, thparams, t_before_training):

            loss_history, accuracy_history = metrics
            lr, minibatch_size = thparams

            training_information = \
            {
                'model_params': self.model.state_dict(),
                'vocab': self.data.vocab,
                'data_info':
                {
                    'n_train': self.data.n_train,
                    'n_val': self.data.n_val
                },
                'metrics_history':
                {
                    'loss':
                    {
                        'train': loss_history[0],
                        'val': loss_history[1]
                    },
                    'accuracy':
                    {
                        'train': accuracy_history[0],
                        'val': accuracy_history[1]
                    }
                },
                'training_hparams':
                {
                    'epoch': epoch,
                    'learning_rate': lr,
                    'minibatch_size': minibatch_size
                },
                'delta_t': time()-t_before_training
            }

            return training_information

        ## Scheduled backup - model thread
        if epoch in self.scheduled_checkpoints:
            training_information = get_training_information(epoch, metrics, thparams, t_before_training)
            training_scheduled_backup_path = self.training_dir_path + self.model.name + '_ep' + str(epoch) + self.training_format
            torch.save(training_information, training_scheduled_backup_path)
            figure_.fig.savefig(self.training_dir_path + self.model.name + '_ep' + str(epoch))

        ## Latest frequent backups
        if (epoch != 0) and ((epoch % self.bkp_freq) == 0):
            training_information = get_training_information(epoch, metrics, thparams, t_before_training)
            live_training_backup_path = self.training_dir_path + self.model.name + '_live_ep' + str(epoch) + self.training_format
            prev_live_training_backup_path = self.training_dir_path + self.model.name + '_live_ep' + str(epoch-self.bkp_freq) + self.training_format
            if os.path.exists(prev_live_training_backup_path):
                os.remove(prev_live_training_backup_path)
            if os.path.exists(self.training_dir_path + self.model.name + '_live_ep' + str(epoch-self.bkp_freq) + '.png'):
                os.remove(self.training_dir_path + self.model.name + '_live_ep' + str(epoch-self.bkp_freq) + '.png')
            torch.save(training_information, live_training_backup_path)
            figure_.fig.savefig(self.training_dir_path + self.model.name + '_live_ep' + str(epoch))