import json
import os
import time

import numpy as np
from datetime import datetime
import torch
from torch.optim import SGD
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataset.dataset_fixed_size import MOVEDatasetFixed
from dataset.dataset_full_size import MOVEDatasetFull
from models.model_move import MOVEModel
from models.model_vgg import VGGModel
from models.model_vgg_v2 import VGGModelV2
from models.model_move_nt import MOVEModelNT
from model_evaluate import test
from model_losses import triplet_loss_mining
from utils.utils import average_precision
from utils.utils import import_dataset_from_pt
from utils.utils import triplet_mining_collate
from torch.utils.tensorboard import SummaryWriter

def train_triplet_mining(model, optimizer, train_loader, margin, norm_dist=1, mining_strategy=2):
    """
    Training loop for one epoch
    :param model: model to be trained
    :param optimizer: optimizer for training
    :param train_loader: dataloader for training
    :param margin: margin for the triplet loss
    :param norm_dist: whether to normalize distances by the embedding size
    :param mining_strategy: which online mining strategy to use
    :return: training loss of the current epoch
    """
    model.train()  # setting the model to training mode
    loss_log = []  # initialize the list for logging loss values of each mini-batch

    for batch_idx, batch in enumerate(train_loader):  # training loop
        items, labels = batch
        if torch.cuda.is_available():  # sending the pcp features and the labels to cuda if available
            items = items.cuda()
        res_1 = model(items)  # obtaining the embeddings of each song in the mini-batch
        # calculating the loss value of the mini-batch
        loss = triplet_loss_mining(res_1, model, labels,
                                   margin=margin, mining_strategy=mining_strategy, norm_dist=norm_dist)

        # setting gradients of the optimizer to zero
        optimizer.zero_grad()

        # calculating gradients with backpropagation
        loss.backward()

        # updating the weights
        optimizer.step()

        # logging the loss value of the current mini-batch
        loss_log.append(loss.cpu().item())

    train_loss = np.mean(np.array(loss_log))  # averaging the loss values of each mini-batch

    return train_loss


def validate_triplet_mining(model_move, val_loader, margin, norm_dist=1, mining_strategy=2):
    """
    validation loop for one epoch
    :param model_move: model to be used for validation
    :param val_loader: dataloader for validation
    :param margin: margin for the triplet loss
    :param norm_dist: whether to normalize distances by the embedding size
    :param mining_strategy: which online mining strategy to use
    :return: validation loss of the current epoch
    """
    with torch.no_grad():  # deactivating gradient tracking for testing
        model_move.eval()  # setting the model to evaluation mode
        loss_log = []  # initialize the list for logging loss values of each mini-batch

        for batch_idx, batch in enumerate(val_loader):  # training loop
            items, labels = batch

            if torch.cuda.is_available():  # sending the pcp features and the labels to cuda if available
                items = items.cuda()

            res_1 = model_move(items)  # obtaining the embeddings of each song in the mini-batch

            # calculating the loss value of the mini-batch
            loss = triplet_loss_mining(res_1, model_move, labels,
                                       margin=margin, mining_strategy=mining_strategy, norm_dist=norm_dist)

            # logging the loss value of the current mini-batch
            loss_log.append(loss.cpu().item())

        val_loss = np.mean(np.array(loss_log))  # averaging the loss values of each mini-batch

    return val_loss

def train(defaults, save_name, dataset_name):
    """
    Main training function of MOVE. For a detailed explanation of parameters,
    please check 'python move_main.py -- help'
    :param save_name: name to save model and experiment summary
    :param train_path: path of the training data
    :param chunks: how many chunks to use for the training data
    :param val_path: path of the validation data
    :param save_model: whether to save model (1) or not (0)
    :param save_summary: whether to save experiment summary (1) or not (0)
    :param seed: random seed
    :param num_of_epochs: number of epochs for training
    :param model_type: which model to use: MOVE (0) or MOVE without transposition invariance (1)
    :param emb_size: the size of the final embeddings produced by the model
    :param sum_method: the summarization method for the model
    :param final_activation: final activation to use for the model
    :param lr: value of learning rate
    :param lrsch: which learning rate scheduler to use
    :param lrsch_factor: the decrease rate of learning rate
    :param momentum: momentum for optimizer
    :param patch_len: number of frames for each song to be used in training
    :param num_of_labels: number of labels per mini-batch
    :param ytc: whether to exclude the songs overlapping with ytc for training
    :param data_aug: whether to use data augmentation
    :param norm_dist: whether to normalize squared euclidean distances with the embedding size
    :param mining_strategy: which mining strategy to use
    :param margin: the margin for the triplet loss
    """
    d = defaults
    train_path = os.path.join(d['dataset_root'], dataset_name + '_train')
    val_path = os.path.join(d['dataset_root'], dataset_name + '_val.pt')

    summary = dict()  # initializing the summary dict
    writer = SummaryWriter(f'runs/{datetime.now().strftime("%m-%d_%H:%M:%S")}-{dataset_name}')
    # dataset_name =

    # initiating the necessary random seeds
    np.random.seed(d['random_seed'])
    torch.manual_seed(d['random_seed'])

    # initializing the model
    model = VGGModelV2(emb_size=256)

    # sending the model to gpu, if available
    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(d['random_seed'])

    # initiating the optimizer
    optimizer = SGD(model.parameters(),
                    lr=d['learning_rate'],
                    momentum=d['momentum'])

    # initializing the lists for tracking losses
    train_loss_log = []
    val_loss_log = []
    val_map_log = []

    # loading the training and validation data
    if d['chunks'] == 1:  # hack for handling 1 chunk for training data
        train_path = '{}_1.pt'.format(train_path)
    else:
        train_path = train_path
    train_data, train_labels = import_dataset_from_pt('{}'.format(train_path), chunks=d['chunks'])
    print('Train data has been loaded!')

    val_data, val_labels = import_dataset_from_pt('{}'.format(val_path), chunks=1)
    print('Validation data has been loaded!')

    # selecting the H dimension of the input data
    # different models handle different size inputs
    patch_len, h = 128, 128

    # initializing the MOVE dataset objects and data loaders
    # we use validation set to track two things, (1) triplet loss, (2) mean average precision
    # to check mean average precision on the full songs,
    # we need to define another dataset object and data loader for it
    train_set = MOVEDatasetFixed(train_data, train_labels, h=h, w=patch_len,
                                 data_aug=d['data_aug'])
    train_loader = DataLoader(train_set, batch_size=d['num_of_labels'], shuffle=True,
                              collate_fn=triplet_mining_collate, drop_last=True)
    val_set = MOVEDatasetFixed(val_data, val_labels, h=h, w=patch_len, data_aug=0)
    val_loader = DataLoader(val_set, batch_size=d['num_of_labels'], shuffle=True,
                            collate_fn=triplet_mining_collate, drop_last=True)
    val_map_set = MOVEDatasetFull(val_data, val_labels)
    val_map_loader = DataLoader(val_map_set, batch_size=1, shuffle=False)

    # initializing the learning rate scheduler
    if d['lr_schedule'] == 0:
        pass
    else:
        if d['lr_schedule'] == 1:
            milestones = [80]
        else:
            milestones = [80, 100]
        lr_schedule = lr_scheduler.MultiStepLR(optimizer,
                                               milestones=milestones,
                                               gamma=d['lrsch_factor'])

    # calculating the number of parameters of the model
    tmp = 0
    for p in model.parameters():
        tmp += np.prod(p.size())
    print('Num of parameters = {}'.format(int(tmp)))

    print('--- Training starts ---')
    print('Model name: {}'.format(save_name))

    start_time = time.monotonic()  # start time for tracking the duration of entire training

    # main training loop
    for epoch in range(d['num_of_epochs']):
        last_epoch = epoch  # tracking last epoch to make sure that model didn't quit early

        start = time.monotonic()  # start time for the training loop
        train_loss = train_triplet_mining(model=model,
                                          optimizer=optimizer,
                                          train_loader=train_loader,
                                          margin=d['margin'],
                                          norm_dist=d['norm_dist'],
                                          mining_strategy=d['mining_strategy'])
        print('Training loop: Epoch {} - Duration {:.2f} mins'.format(epoch, (time.monotonic()-start)/60))

        start = time.monotonic()  # start time for the validation loop
        val_loss = validate_triplet_mining(model_move=model,
                                           val_loader=val_loader,
                                           margin=d['margin'],
                                           norm_dist=d['norm_dist'],
                                           mining_strategy=d['mining_strategy'])

        print('Validation loop: Epoch {} - Duration {:.2f} mins'.format(epoch, (time.monotonic()-start)/60))

        start = time.monotonic()  # start time for the mean average precision calculation

        # calculating the pairwise distances on validation set
        dist_map_matrix = test(model=model,
                               test_loader=val_map_loader).cpu()

        # calculation performance metrics
        # average_precision function uses similarities, not distances
        # we multiple the distances with -1, and set the diagonal (self-similarity) -inf
        val_map_score = average_precision(
            os.path.join(d['dataset_root'], f'ytrue_val_{dataset_name}.pt'),
            -1 * dist_map_matrix.float().clone() + torch.diag(torch.ones(len(val_data)) * float('-inf')),
        )
        print('Test loop: Epoch {} - Duration {:.2f} mins'.format(epoch, (time.monotonic()-start)/60))

        # saving loss values for the summary
        train_loss_log.append(train_loss)
        val_loss_log.append(val_loss)
        val_map_log.append(val_map_score.item())

        # saving model if needed
        if d['save_model'] == 1:
            if not os.path.exists('saved_models/'):
                os.mkdir('saved_models/')
            torch.save(model.state_dict(), 'saved_models/model_{}.pt'.format(save_name))

        # printing the losses
        print('training_loss: {}'.format(train_loss))
        print('val_loss: {}'.format(val_loss))

        # activate learning rate scheduler if needed
        if d['lr_schedule'] != 0:
            lr_schedule.step()

        # dumping current loss values to the summary
        summary['train_loss_log'] = train_loss_log
        summary['val_loss_log'] = val_loss_log
        summary['val_map_log'] = val_map_log
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('mAP/Val', val_map_score, epoch)

    end_time = time.monotonic()  # end time of the entire training loop
