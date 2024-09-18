import numpy as np
import torch
import torch.utils.data

from transformer import Constants
import pandas as pd


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data, config):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.event_type = [[elem['type_event'] + 1 for elem in inst] for inst in data]

        self.covariates = [[elem['covariates'] for elem in inst] for inst in data]

        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.time_gap[idx], self.event_type[idx], self.covariates[idx] #, self.next_event_time[idx], self.next_event_type[idx]


def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)

def pad_covariates(insts):
    batch = len(insts)
    max_len = max(len(inst) for inst in insts)

    d_covar = insts[0][0].shape[-1]
    result = []

    for idx in range(batch):
        seq = insts[idx]

        tmp = [torch.tensor(np.array(inst.iloc[-1]).astype(np.float32), dtype=torch.float32) for inst in seq] + \
            [torch.ones((d_covar))*Constants.PAD] * (max_len - len(seq))

        result.append(tmp)



    return result
def padd_covariates(insts):
    batch = len(insts)
    max_len = max(len(inst) for inst in insts)
    d_covar = insts[0][0].shape[-1]
    result = []
    for idx in range(batch):
        tmp = [torch.tensor(inst, dtype=torch.float32) for inst in insts[idx]] + \
            [torch.ones((1, d_covar))*Constants.PAD
             ]*(max_len - len(insts[idx]))
        result.append(tmp)


    return result



def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    time, time_gap, event_type, covariates = list(zip(*insts))


    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)
    covariates_ = pad_covariates(covariates)



    return time, time_gap, event_type, covariates_ #covariates


def get_dataloader(data, config, shuffle=True):
    """ Prepare dataloader. """

    ds = EventData(data, config)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl
