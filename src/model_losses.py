import torch
import torch.nn.functional as F

from utils.utils import pairwise_distance_matrix


def triplet_loss_mining(res_1, labels, embedding_size,
                        margin=1, mining_strategy='hard', norm_dist=True, indices=None, margin_adapter=None):
    """
    Online mining function for selecting the triplets
    :param res_1: embeddings in the mini-batch
    :param labels: labels of the embeddings
    :param embedding_size: Size of the output embedding
    :param margin: margin for the triplet loss
    :param mining_strategy: which mining strategy to use (0 for random, 1 for semi-hard, 2 for hard)
    :param norm_dist: whether to normalize the distances by the embedding size
    :return: triplet loss value
    """

    # Creating positive and negative masks for online mining
    i_labels = []
    column_labels = []
    for i, l in enumerate(labels):
        if l in column_labels:
            duplicated_label_index = column_labels.index(l)
            duplicated_index = i_labels[duplicated_label_index]
            i = duplicated_index
        i_labels += [i] * 4
        column_labels += [l] * 4

    i_labels = torch.Tensor(i_labels).view(-1, 1)

    mask_diag = (1 - torch.eye(res_1.size(0))).long()
    if torch.cuda.is_available():
        i_labels = i_labels.cuda()
        mask_diag = mask_diag.cuda()
    temp_mask = (pairwise_distance_matrix(i_labels) < 0.5).long()

    mask_pos = mask_diag * temp_mask
    mask_neg = mask_diag * (1 - mask_pos)

    # Getting the pairwise distance matrix
    dist_all = pairwise_distance_matrix(res_1)
    # Normalizing the distances by the embedding size
    if norm_dist:
        dist_all /= embedding_size

    if mining_strategy == 'random':  # Random mining
        dists_pos, dists_neg, sel_pos, sel_neg = triplet_mining_random(dist_all, mask_pos, mask_neg)
    elif mining_strategy == 'semihard':  # Semi-hard mining
        dists_pos, dists_neg, sel_pos, sel_neg = triplet_mining_semihard(dist_all, mask_pos, mask_neg)
    else:  # Hard mining by default
        dists_pos, dists_neg, sel_pos, sel_neg = triplet_mining_hard(dist_all, mask_pos, mask_neg)

    # Adapt margin based on the selected labels
    if margin_adapter:
        margin = margin_adapter.adapt(column_labels, sel_pos, sel_neg)
    else:
        margin_list = [[margin] for _ in range(dists_pos.size(0))]
        margin = torch.tensor(margin_list).cuda()

    hard_indices = []
    if indices is not None:
        hard_indices = find_hard_indices(dist_all, indices, mask_pos, mask_neg)

    margin_satisfied = 0
    # Loss = max(Distance_anc_pos - Distance_anc_neg + Margin, 0)

    dists = dists_pos - dists_neg
    # loss = F.relu(torch.where(dists > 1, dists ** 2, dists) + margin)

    loss = F.relu(dists_pos - dists_neg + margin)  # calculating triplet loss
    for pos, neg, m in zip(dists_pos, dists_neg, margin):
        if pos - neg + m  <= 0:
            margin_satisfied += 1

    margin_satisfied_rate = margin_satisfied / len(dists_pos)

    return loss.mean(), dists_pos.mean(), dists_neg.mean(), margin_satisfied_rate, hard_indices


def find_hard_indices(dist_all, indices, mask_pos, mask_neg):
    """
    Find the indices of the examples which are hardest in a batch
    :param dist_all:
    :param indices:
    :return:
    """
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    # Selecting the positive elements of triplets
    _, sel_pos = torch.max(dist_all * mask_pos.double(), 1)

    # Modifying the negative mask for hard mining
    mask_neg = torch.where(mask_neg == 0, torch.tensor(float('inf'), device=device), torch.tensor(1., device=device))
    # Selecting the negative elements of triplets
    _, sel_neg = torch.min(dist_all + mask_neg.double(), 1)
    hard_sel = torch.cat((sel_pos.view(-1, 1), sel_neg.view(-1, 1)), dim=1).tolist()
    hard_indices = [(indices[pos], indices[neg]) for pos, neg in hard_sel]
    return hard_indices


def triplet_mining_hard(dist_all, mask_pos, mask_neg):
    """
    Performs online hard triplet mining (both positive and negative)
    :param dist_all: pairwise distance matrix
    :param mask_pos: mask for positive elements of triplets
    :param mask_neg: mask for negative elements of triplets
    :return: selected positive and negative distances
    """
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # Selecting the positive elements of triplets
    _, sel_pos = torch.max(dist_all * mask_pos.double(), 1)
    dists_pos = torch.gather(dist_all, 1, sel_pos.view(-1, 1))
    # Modifying the negative mask for hard mining
    mask_neg = torch.where(mask_neg == 0, torch.tensor(float('inf'), device=device), torch.tensor(1., device=device))

    # Selecting the negative elements of triplets
    _, sel_neg = torch.min(dist_all + mask_neg.double(), 1)
    dists_neg = torch.gather(dist_all, 1, sel_neg.view(-1, 1))

    return dists_pos, dists_neg, sel_pos, sel_neg


def triplet_mining_random(dist_all, mask_pos, mask_neg):
    """
    Performs online random triplet mining
    :param dist_all: pairwise distance matrix
    :param mask_pos: mask for positive elements of triplets
    :param mask_neg: mask for negative elements of triplets
    :return: selected positive and negative distances
    """
    # selecting the positive elements of triplets
    _, sel_pos = torch.max(mask_pos.double() + torch.rand_like(dist_all), 1)
    dists_pos = torch.gather(dist_all, 1, sel_pos.view(-1, 1))

    # selecting the negative elements of triplets
    _, sel_neg = torch.max(mask_neg.double() + torch.rand_like(dist_all), 1)
    dists_neg = torch.gather(dist_all, 1, sel_neg.view(-1, 1))

    return dists_pos, dists_neg, sel_pos, sel_neg


def triplet_mining_semihard(dist_all, mask_pos, mask_neg):
    """
    Performs online semi-hard triplet mining (a random positive, a semi-hard negative)
    :param dist_all: pairwise distance matrix
    :param mask_pos: mask for positive elements of triplets
    :param mask_neg: mask for negative elements of triplets
    :return: selected positive and negative distances
    """
    # Selecting the positive elements of triplets
    _, sel_pos = torch.max(mask_pos.float() + torch.rand_like(dist_all), 1)
    dists_pos = torch.gather(dist_all, 1, sel_pos.view(-1, 1))

    # Selecting the negative elements of triplets
    _, sel_neg = torch.max((mask_neg + mask_neg * (dist_all < dists_pos.expand_as(dist_all)).long()).float() +
                           torch.rand_like(dist_all), 1)
    dists_neg = torch.gather(dist_all, 1, sel_neg.view(-1, 1))

    return dists_pos, dists_neg, sel_pos, sel_neg

