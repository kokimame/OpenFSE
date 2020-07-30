import json
import os
import time

import numpy as np
from datetime import datetime
import torch
from torch.optim import SGD
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataset.dataset_fixed_size import DatasetFixed
from dataset.dataset_full_size import DatasetFull
from dataset.train_loader import TrainLoader
from models.model_move import MOVEModel
from models.model_vgg import VGGModel
from models.model_vgg_v2 import VGGModelV2
from models.model_vgg_dropout import VGGModelDropout
from models.model_move_nt import MOVEModelNT
from model_evaluate import test
from model_losses import triplet_loss_mining
from margin_adapter import MarginAdapter

from utils.utils import average_precision
from utils.utils import import_dataset_from_pt
from utils.utils import triplet_mining_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

def train_triplet_mining(model, optimizer, train_loader, margin,
                         norm_dist=True, mining_strategy='hard', margin_adapter=None):
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
    pos_log = []
    neg_log = []
    msr_log = []

    for batch in tqdm(train_loader, desc='Training the model .....'):  # training loop
        items, item_info = batch
        labels, sound_idxs = [], []
        for label, some_idx in item_info:
            labels.append(label)
            sound_idxs.extend(some_idx)
        if torch.cuda.is_available():
            items = items.cuda()
        output = model(items)  # obtaining the embeddings of each song in the mini-batch
        # calculating the loss value of the mini-batch
        loss, pos_avg, neg_avg, msr = triplet_loss_mining(
            output, labels, model.fin_emb_size,
            margin=margin, mining_strategy=mining_strategy,
            norm_dist=norm_dist, indices=sound_idxs, margin_adapter=margin_adapter
        )
        # setting gradients of the optimizer to zero
        optimizer.zero_grad()

        # calculating gradients with backpropagation
        loss.backward()

        # updating the weights
        optimizer.step()

        # logging the loss value of the current mini-batch
        loss_log.append(loss.cpu().item())
        pos_log.append(pos_avg.cpu().item())
        neg_log.append(neg_avg.cpu().item())
        msr_log.append(msr)

    train_loss = np.mean(np.array(loss_log))  # averaging the loss values of each mini-batch
    train_pos = np.mean(np.array(pos_log))
    train_neg = np.mean(np.array(neg_log))
    train_msr = np.mean(np.array(msr_log))

    return train_loss, train_pos, train_neg, train_msr


def validate_triplet_mining(model, val_loader, margin,
                            norm_dist=True, mining_strategy='hard', margin_adapter=None):
    """
    validation loop for one epoch
    :param model: model to be used for validation
    :param val_loader: dataloader for validation
    :param margin: margin for the triplet loss
    :param norm_dist: whether to normalize distances by the embedding size
    :param mining_strategy: which online mining strategy to use
    :return: validation loss of the current epoch
    """
    with torch.no_grad():  # deactivating gradient tracking for testing
        model.eval()  # setting the model to evaluation mode
        loss_log = []  # initialize the list for logging loss values of each mini-batch
        msr_log = []
        pos_log = []
        neg_log = []

        for batch_idx, batch in enumerate(val_loader):  # training loop
            items, item_info = batch
            labels, sound_idxs = [], []
            for label, some_idx in item_info:
                labels.append(label)
                sound_idxs.extend(some_idx)

            if torch.cuda.is_available():  # sending the pcp features and the labels to cuda if available
                items = items.cuda()

            res_1 = model(items)  # obtaining the embeddings of each song in the mini-batch

            # calculating the loss value of the mini-batch
            loss, pos_avg, neg_avg, msr = triplet_loss_mining(res_1, labels, model.fin_emb_size,
                                                                 margin=margin,
                                                                 mining_strategy=mining_strategy,
                                                                 indices=sound_idxs,
                                                                 norm_dist=norm_dist)

            # logging the loss value of the current mini-batch
            loss_log.append(loss.cpu().item())
            pos_log.append(pos_avg.cpu().item())
            neg_log.append(neg_avg.cpu().item())
            msr_log.append(msr)

        val_loss = np.mean(np.array(loss_log))  # averaging the loss values of each mini-batch
        val_pos = np.mean(np.array(pos_log))
        val_neg = np.mean(np.array(neg_log))
        val_msr = np.mean(np.array(msr_log))  # averaging the loss values of each mini-batch

    return val_loss, val_pos, val_neg, val_msr

def train(defaults, save_name, dataset_name):
    """
    Main training function. For a detailed explanation of parameters,
    please check 'python move_main.py -- help'
    """
    print(f'Start training {datetime.now()}')
    d = defaults
    train_path = os.path.join(d['dataset_root'], dataset_name + '_train')
    val_path = os.path.join(d['dataset_root'], dataset_name + '_val.pt')

    # initiating the necessary random seeds
    np.random.seed(d['random_seed'])
    torch.manual_seed(d['random_seed'])

    if not os.path.exists('saved_models/'):
        os.mkdir('saved_models/')

    # initializing the model
    model = VGGModelDropout(emb_size=d['emb_size'])
    if d['use_pretrained']:
        model.load_state_dict(torch.load(d['use_pretrained']))
        save_name = Path(d['use_pretrained']).stem.replace('model_', '')
        _, yyyymmdd, hhmmssDot = save_name.rsplit('_', 2)
        yyyy, mm, dd = yyyymmdd.split('-')
        hour, minute, secDot = hhmmssDot.split(':')
        sec = secDot.split('.')[0]
        run_file = f'runs/{mm}-{dd}_{hour}-{minute}-{sec}-{dataset_name}'
        writer = SummaryWriter(run_file)
        print(f'Save name updated to {save_name}')
        print(f'Run file updated to {run_file}')
    else:
        writer = SummaryWriter(f'runs/{datetime.now().strftime("%m-%d_%H-%M-%S")}-{dataset_name}')


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

    # loading the training and validation data
    if d['chunks'] == 1:  # hack for handling 1 chunk for training data
        train_path = '{}_1.pt'.format(train_path)
    else:
        train_path = train_path
    train_data, train_labels, train_ids = import_dataset_from_pt('{}'.format(train_path), chunks=d['chunks'])

    print(f'Train data has been loaded! Length: {len(train_data)}')

    val_data, val_labels, val_ids = import_dataset_from_pt('{}'.format(val_path), chunks=1)
    print('Validation data has been loaded!')

    # Initialize the dataset objects and data loaders
    # we use validation set to track two things, (1) triplet loss, (2) mean average precision
    # to check mean average precision on the full sounds,
    # we need to define another dataset object and data loader for it
    train_set = DatasetFixed(train_data, train_labels, train_ids, h=d['input_height'], w=d['input_width'])# , data_aug=d['data_aug'])
    train_loader = TrainLoader(train_set, batch_size=d['num_of_labels'], shuffle=True,
                              collate_fn=triplet_mining_collate, drop_last=True)

    val_set = DatasetFixed(val_data, val_labels, val_ids, h=d['input_height'], w=d['input_width'])#, data_aug=d['data_aug'])
    val_loader = DataLoader(val_set, batch_size=d['num_of_labels'], shuffle=True,
                            collate_fn=triplet_mining_collate, drop_last=True)

    if d["adaptive_margin"]:
        margin_adapter = MarginAdapter([train_labels, val_labels],
                                       base_margin=d['margin'], description_file='data/AudioSet_lookup.json')
    else:
        margin_adapter = None
    print(f'Adaptive margin: {"on" if margin_adapter else "off"}')
    print(f'Mining strategy: {d["mining_strategy"]}')

    # Validation dataset to compute mAP
    val_mAP_set = DatasetFull(val_data, val_labels)
    val_mAP_loader = DataLoader(val_mAP_set, batch_size=8, shuffle=False)

    # Initializing the learning rate scheduler
    if d['lr_milestones'] is not None:
        lr_schedule = lr_scheduler.MultiStepLR(optimizer,
                                               milestones=d['lr_milestones'],
                                               gamma=d['lrsch_factor'])

    # Calculating the number of parameters of the model
    tmp = 0
    for p in model.parameters():
        tmp += np.prod(p.size())
    print('Num of parameters = {}'.format(int(tmp)))

    print('--- Training starts ---')
    print('Model name: {}'.format(save_name))

    # Main training loop
    for epoch in range(d['num_of_epochs']):
        time.sleep(0.01)
        train_loss, train_pos, train_neg, train_msr = train_triplet_mining(model=model,
                                                                           optimizer=optimizer,
                                                                           train_loader=train_loader,
                                                                           margin=d['margin'],
                                                                           norm_dist=d['norm_dist'],
                                                                           mining_strategy=d['mining_strategy'],
                                                                           margin_adapter=margin_adapter)

        val_loss, val_pos, val_neg, val_msr = validate_triplet_mining(model=model,
                                                                      val_loader=val_loader,
                                                                      margin=d['margin'],
                                                                      norm_dist=d['norm_dist'],
                                                                      mining_strategy=d['mining_strategy'],
                                                                      margin_adapter=margin_adapter)

        # saving model if needed
        if d['save_model']:
            torch.save(model.state_dict(), 'saved_models/model_{}.pt'.format(save_name))

        # Activate learning rate scheduler if needed
        if d['lr_milestones'] is not None:
            lr_schedule.step()

        # dumping current loss values to the summary
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalars('Train/Distance', {'pos': train_pos, 'neg': train_neg}, epoch)
        writer.add_scalar('Train/Margin Satisfied (%)', train_msr * 100, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalars('Val/Distance', {'pos': val_pos, 'neg': val_neg}, epoch)
        writer.add_scalar('Val/Margin Satisfied (%)', val_msr * 100, epoch)
        print(f'Epoch: {epoch}\n'
              f'Train result: Loss({train_loss:.2f})\n'
              f'Avg. Dist.(P:{train_pos:.2f}|N:{train_neg:.2f}|P-N:{train_pos - train_neg:.2f})\n'
              f'Margin Satisfied (%)({int(train_msr * 100)})')
        # Calculation performance metrics
        # average_precision function uses similarities, not distances
        # we multiple the distances with -1, and set the diagonal (self-similarity) -inf
        if epoch % d['test_per_epoch'] == 0:
            # calculating the pairwise distances on validation set
            dist_map_matrix = test(model=model,
                                   test_loader=val_mAP_loader).cpu()

            mAP, mrr, mr, top1, top10, ones_avg = average_precision(
                os.path.join(d['dataset_root'], f'{dataset_name}_val_ytrue.pt'),
                -1 * dist_map_matrix.float().clone() + torch.diag(torch.ones(len(val_data)) * float('-inf')),
                k=d['mAP@k']
            )
            # writer.add_scalar('Test/Top10', top10, epoch)
            # writer.add_scalar('Test/Top1', top1, epoch)
            # writer.add_scalar('Test/MR', mr, epoch)
            # writer.add_scalar('Test/MRR', mrr, epoch)
            writer.add_scalar('Test/mAP', mAP, epoch)
            writer.add_scalar('Test/1sAvg', ones_avg, epoch)
