import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision.models import resnet34
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import os
import time
import shutil
import numpy as np
import itertools
import matplotlib.pyplot as plt
import torch.nn as nn
import cv2
from tqdm import tqdm
from utils import *
from sklearn.metrics import confusion_matrix
import csv
import random
from sklearn.metrics import classification_report
import torchvision
from models.ResNet34.model import Model


class Trainer(object):

    def __init__(self, config, trainloader, validloader, testloader, iteration_value):
        self.config = config
        self.random_seed = iteration_value
        # data params
        self.train_loader = trainloader
        self.valid_loader = validloader
        self.test_loader = testloader

        self.num_train = self.train_loader.sampler.num_samples
        self.num_valid = self.valid_loader.sampler.num_samples
        self.num_test = self.test_loader.sampler.num_samples

        self.num_classes = config.nb_classes
        self.num_channels = 3

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.lr = config.init_lr

        
        self.model_name = 'ResNet34'
        self.use_gpu = config.use_gpu
        self.best = config.best
        self.dataset_name = 'CrohnIPI_consensus'
        if config.test:
            self.ckpt_dir = os.path.join('../test/ckpt', self.model_name, self.dataset_name)
        else:
            self.ckpt_dir = os.path.join('../ckpt', self.model_name, self.dataset_name)
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.resume = config.resume

        self.epoch_val =0


        self.model = Model(self.num_classes)


        if self.use_gpu:
            self.model.cuda()

        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))


        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)


    def train(self):
        """
        Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
        )
        print(self.model_name)
        for epoch in range(self.start_epoch, self.epochs):

            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch + 1, self.epochs, self.lr)
            )

            # train for 1 epoch
            train_loss, train_acc = self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_loss, valid_acc = self.validate(epoch)



            is_best = valid_acc > self.best_valid_acc
            msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f}"
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(msg.format(train_loss, train_acc, valid_loss, valid_acc))

            # check for improvement
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                break
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {'epoch': epoch + 1,
                 'model_state': self.model.state_dict(),
                 'optim_state': self.optimizer.state_dict(),
                 'best_valid_acc': self.best_valid_acc,
                 }, is_best
            )

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y, path) in enumerate(self.train_loader):
                if self.use_gpu:
                    x, y = x.cuda(), y.cuda()
                x, y = Variable(x), Variable(y)

                self.batch_size = x.shape[0]
                # last iteration
                pred = self.model(x)
                pred = F.log_softmax(pred, dim=1)

                predicted = torch.max(pred, 1)[1]

                # compute loss
                loss = F.nll_loss(pred, y)

                # compute accuracy
                correct = (predicted == y).float()
                acc = 100 * (correct.sum() / len(y))

                # store
                losses.update(loss.item(), x.size()[0])
                accs.update(acc.item(), x.size()[0])

                # compute gradients
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                            (toc - tic), loss.item(), acc.item()
                        )
                    )
                )
                pbar.update(self.batch_size)

            return losses.avg, accs.avg

    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()

        for i, (x, y, path) in enumerate(self.valid_loader):
            if self.use_gpu:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]

            pred = self.model(x)
            pred = F.log_softmax(pred, dim=1)
            # calculate reward
            predicted = torch.max(pred, 1)[1]

            # compute losses for differentiable modules
            loss_action = F.nll_loss(pred, y)

            # sum up into a hybrid loss
            loss = loss_action

            # compute accuracy
            correct = (predicted == y).float()
            acc = 100 * (correct.sum() / len(y))

            # store
            losses.update(loss.item(), x.size()[0])
            accs.update(acc.item(), x.size()[0])
        return losses.avg, accs.avg

    def save_checkpoint(self, state, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        # print("[*] Saving model to {}".format(self.ckpt_dir))


        filename = self.model_name + '_' + self.dataset_name + '_' + str(self.random_seed) +'_'+str(self.epoch_val )+ '_ckpt.pth.tar'
        if not os.path.exists(os.path.join(self.ckpt_dir, 'last_epoch')):
            os.makedirs(os.path.join(self.ckpt_dir, 'last_epoch'))
        ckpt_path = os.path.join(self.ckpt_dir, 'last_epoch', filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.model_name + '_' + self.dataset_name + '_' + str(self.random_seed) + '_model_best.pth.tar'
            if not os.path.exists(os.path.join(self.ckpt_dir, 'best')):
                os.makedirs(os.path.join(self.ckpt_dir, 'best'))
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, 'best', filename)
            )
    def transform(multilevelDict):
        return
    def load_checkpoint(self, best=False):
        """
        Load the best copy of a model. This is useful for 2 cases:

        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Params
        ------
        - best: if set to True, loads the best model. Use this if you want
          to evaluate your model on the test data. Else, set to False in
          which case the most recent version of the checkpoint is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))



        if best:
            filename = self.model_name + '_' + self.dataset_name + '_' + str(self.random_seed) + '_model_best.pth.tar'
            ckpt_path = os.path.join(self.ckpt_dir, 'best', filename)
        else:
            filename = self.model_name + '_' + self.dataset_name + '_' + str(self.random_seed) + '_ckpt.pth.tar'
            ckpt_path = os.path.join(self.ckpt_dir, 'last_epoch', filename)
        ckpt = torch.load(ckpt_path)

        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])


        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt['epoch'], ckpt['best_valid_acc'])
            )
        else:
            print(
                "[*] Loaded {} checkpoint @ epoch {}".format(
                    filename, ckpt['epoch'])
            )

