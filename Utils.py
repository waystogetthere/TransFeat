import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import get_non_pad_mask


def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def compute_event(event, non_pad_mask):
    """ Log-likelihood of events. """

    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result


def compute_integral_biased(all_lambda, time, non_pad_mask):
    """ Log-likelihood of non-events, using linear interpolation. """

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask[:, 1:]

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result


def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 10

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    # [BATCH, LEN-1], the elapsed time since last event

    temp_time = diff_time.unsqueeze(2) * torch.rand([*diff_time.size(), num_samples], device=data.device)
    # for each duration d, sample 100 random timestamps in [0,d]

    temp_time /= (time[:, :-1] + 1).unsqueeze(2)
    # divide by last event time

    temp_hid = model.linear(data)[:, :-1, :]
    # use the 'last hidden state' instead of 'current hidden state'.
    num_types = temp_hid.shape[2]

    temp_time = temp_time.unsqueeze(-2)
    temp_time = temp_time.repeat(1, 1, num_types, 1)
    # expand a new axis at -2, now temp_time.shape: [BATCH, LEN, Num_types, Num_samples]

    # temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True) # the old code

    temp_hid = temp_hid.unsqueeze(-1)
    temp_hid = temp_hid.repeat(1, 1, 1, num_samples)
    # expand a new axis at -1, now temp_hid.shape: [BATCH, LEN, Num_types, Num_samples]

    all_lambda = softplus(temp_hid + model.alpha * temp_time, model.beta)
    all_lambda = torch.sum(all_lambda, dim=3) / num_samples
    # Sum all along the Num_samples axis, get the average intensity during each duration for each types

    unbiased_integral = all_lambda * diff_time.unsqueeze(-1).repeat(1,1,num_types)
    # expand and repeat the diff_time from [BATCH, LEN] to [BATCH, LEN, Num_types]

    unbiased_integral = torch.sum(unbiased_integral, dim=-1) # sum along the Num_types axis
    unbiased_integral = torch.sum(unbiased_integral, dim=-1) # sum along the LEN axis

    return unbiased_integral


def log_likelihood(model, data, time, types):
    """ Log-likelihood of sequence. """

    non_pad_mask = get_non_pad_mask(types).squeeze(2)

    type_mask = torch.zeros([*types.size(), model.num_types], device=data.device)
    for i in range(model.num_types):
        type_mask[:, :, i] = (types == i + 1).bool().to(data.device)

    all_hid = model.linear(data)

    all_lambda = softplus(all_hid, model.beta)

    type_lambda = torch.sum(all_lambda * type_mask, dim=2)

    # event log-likelihood
    event_ll = compute_event(type_lambda, non_pad_mask)
    event_ll = torch.sum(event_ll, dim=-1)

    # non-event log-likelihood, either numerical integration or MC integration
    # non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
    non_event_ll = compute_integral_unbiased(model, data, time, non_pad_mask, type_mask)
    non_event_ll = torch.sum(non_event_ll, dim=-1)

    return event_ll, non_event_ll


def type_loss(prediction, types, loss_func):
    """ Event prediction loss, cross entropy or label smoothing. """

    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    

    truth = types[:, 1:] - 1
    # truth = types[:, :-1] - 1

    prediction = prediction[:, :-1, :]
    # prediction = prediction[:, :-1, :]
    pred_type = torch.max(prediction, dim=-1)[1]

    # assert False
    focal_loss = False
    if focal_loss:
        _, seq_len, num_class = prediction.shape
        haha_truth = truth.reshape(-1)
        haha_mask = haha_truth != -1

        haha_prediction = prediction.reshape(-1, num_class)
        # print('seq_len', seq_len)
        # for i in range(haha_prediction.shape[0]):
        #     seq_idx = i // seq_len
        #     loc = i % seq_len
        #     print('the i-th', i, seq_idx, loc)
        #     # print(haha_prediction[i,:])
            
        #     # print(prediction[seq_idx,loc,:])
        #     assert (prediction[seq_idx,loc,:] == haha_prediction[i,:]).all()
        # assert False
        outputs = haha_prediction[haha_mask, :]
        targets = haha_truth[haha_mask]
        # print(outputs.shape, targets.shape, 'ready for focal!')

        # ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
        weight=torch.Tensor([1-0.035-0.211,1-0.7535-0.211,0.035]).to('cuda')
        ce_loss = torch.nn.functional.cross_entropy(outputs, targets,reduction='mean',weight=weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** 2 * ce_loss).mean()


        pred_type = torch.max(prediction, dim=-1)[1]

        correct_num = torch.sum(pred_type == truth)
        return focal_loss, correct_num


    correct_num = torch.sum(pred_type == truth)
    # print('correct num', correct_num)
    # assert False

    # compute cross entropy loss
    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)
    return loss, correct_num


def time_loss(prediction, time_gap):
    """ Time prediction loss. """

    prediction.squeeze_(-1)

    # true = event_time[:, 1:] - event_time[:, :-1]
    prediction = prediction[:, :-1]
    time_gap = time_gap[:, 1:]
    # event time gap prediction
    diff = prediction - time_gap
    se = torch.sum(diff * diff)
    return se


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss
