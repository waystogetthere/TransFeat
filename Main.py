import argparse
from operator import gt
import numpy as np
import pandas as pd
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

import transformer.Constants as Constants
import Utils

from preprocess.Dataset import get_dataloader
# from transformer.Models import Transformer
from transformer.CoHawkes import Transformer
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix as confusion_matrix


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


MODEL_PATH_BEST = 'best_model.pth'
MODEL_PATH_FINAL = 'final_model.pth'




def prepare_dataloader(config):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')

            num_types = data['dim_process']
            d_covar = data['dim_fea']
            data = data[dict_name]
            return data, int(num_types), int(d_covar)
        
    print('[Info] Loading train data...')
    tr_name = config.tr_name
    test_name = config.test_name

    train_data, num_types, d_covar = load_data(config.data + tr_name, 'train')

    print('[Info] Loading test data...')
    test_data, _, _ = load_data(config.data + test_name, 'test')


    trainloader = get_dataloader(train_data, config, shuffle=True)
    testloader = get_dataloader(test_data, config, shuffle=False)

    return trainloader, testloader, num_types, d_covar


def visual(fig_name_prefix, pred_time, gt_time, err_dict, feature_importance_inner):
    import matplotlib.pyplot as plt




    plt.figure()
    plt.plot(gt_time, label='gt')
    plt.plot(pred_time, label='pred')
    plt.legend()
    plt.title('Pred Time VS GT Time')

    plt.savefig(fig_name_prefix + ' time-plot loss-{:8.5f}.png'.format(err_dict['rmse']))
    plt.close()


    f, ax1 = plt.subplots(1, 2,sharey=True)
    ax1.barh(feature_importance_inner['feature_name'], feature_importance_inner['instance_level'], label='instance-level')
    ax1.set_title('Instance-Level')

    for i, v in enumerate(feature_importance_inner['instance_level'].to_numpy()):
        ax1.text(v + 0.01, i, str(format(v, '.4f')), color='green', fontweight='bold')

    plt.savefig(fig_name_prefix + ' Feature Importance.png', dpi=300)
    plt.close()

    plt.figure()
    plt.plot(feature_importance_inner['instance_level'], marker='o')
    plt.savefig(fig_name_prefix + ' Feature Importance inner-instance-level.png', dpi=300)
    plt.close()


    with open(fig_name_prefix + ' pred_time.npy', 'wb') as f:
        np.save(f, pred_time)
    with open(fig_name_prefix + ' gt_time.npy', 'wb') as f:
        np.save(f, gt_time)







def train_epoch(model, training_data, optimizer, pred_loss_func, epoch, tr_statistics, config):
    """ Epoch operation in training phase. """

    model.train()

    total_loss = 0
    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_hits = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions

    total_event_loss = 0

    Pred_time_total, Next_time_total, Pred_type_total, Next_type_total = ([] for i in range(4))
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        event_time, time_gap, event_type, covariates = batch

        # map from cpu to gpu
        event_time, time_gap, event_type = \
            map(lambda x: x.to(config.device), (event_time, time_gap, event_type))
        for idx, _ in enumerate(covariates):
            covariates[idx] = [inst.to(config.device) for inst in covariates[idx]]

        """ forward """
        _, prediction, _ = model(event_type, event_time, covariates)

        """ backward """

        type_logits, pred_dur, log_prob = prediction
        pred_loss, pred_num_event = Utils.type_loss(type_logits, event_type, pred_loss_func)

        assert not torch.isnan(pred_dur).any()
        se = Utils.time_loss(pred_dur, time_gap)

        mode_1 = len(str(abs(int(log_prob.sum().detach().cpu().item()))))
        mode_2 = len(str(abs(int(pred_loss.detach().cpu().item()))))
        mode_3 = len(str(abs(int(se.detach().cpu().item()))))

        loss = (-1) * log_prob.sum() * (10**(-mode_1)) + pred_loss * (10**(-mode_2+1)) + se * (10**(-mode_3))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
        optimizer.step()

        """ note keeping """
        pred_type = torch.max(type_logits, dim=-1)[1]
        pred_type = pred_type[:, :-1].reshape(-1)
        next_type= event_type[:, 1:].reshape(-1) # flatten
        mask = next_type > 0
        Pred_type_total.append(pred_type[mask])
        Next_type_total.append(next_type[mask])
        
        next_gap = time_gap[:, 1:].reshape(-1)
        pred_dur =  pred_dur[:, :-1].reshape(-1)
        Next_time_total.append(next_gap[mask])
        Pred_time_total.append(pred_dur[mask])


        total_loss += loss
        total_event_ll += log_prob.sum().item()
        total_time_se += se.item()
        total_event_loss += pred_loss.item()
        total_hits += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()

        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]


    ATT = []

    for _, batch in enumerate(training_data):
        """ prepare data """
        event_time, time_gap, event_type, covariates = batch

        # map from cpu to gpu
        event_time, time_gap, event_type = \
            map(lambda x: x.to(config.device), (event_time, time_gap, event_type))
        for idx, _ in enumerate(covariates):
            covariates[idx] = [inst.to(config.device) for inst in covariates[idx]]

        batch, seq_len = len(covariates), len(covariates[0])

        cov_collect = torch.zeros(size=(batch, seq_len, model.d_covar), device=covariates[0][0].device)
        for b_id, seq in enumerate(covariates):
            len_seq = len(seq)
            cov_collect[b_id, :, :] = torch.cat(seq).reshape(len_seq, model.d_covar)
        cov_collect = cov_collect[~torch.all(cov_collect==0, axis=-1)] # eliminate all padding covariates
        att = model.encoder.cov_intra_att.get_attention(cov_collect) 
        ATT.append(att)



    instance_level = torch.mean(torch.cat(ATT), axis=0).detach().cpu().numpy()
    global_attention = model.encoder.cov_intra_att.get_mean_attention_weights().detach().cpu().numpy()

    feature_order = ['feature_{}'.format(i) for i in range(model.d_covar)]


    Fi_inner = pd.DataFrame(columns=['feature_name', 'instance_level'])
    
    Fi_inner['feature_name'] = feature_order
    Fi_inner['instance_level'] = instance_level



    Pred_time_total = [item.item() for sublist in Pred_time_total for item in sublist]
    Next_time_total = [item.item() for sublist in Next_time_total for item in sublist]

    Pred_type_total = [item.item() for sublist in Pred_type_total for item in sublist]
    Next_type_total = [item.item() for sublist in Next_type_total for item in sublist]


    rmse = np.sqrt(total_time_se / total_num_pred)
    mean_event = total_event_ll / total_num_pred  # total_num_event
    acc = total_hits / total_num_pred
    event_loss = total_event_loss/total_num_pred

    err_dict = {
        'total_loss': total_loss,
        'log-likelihood': mean_event,
        'rmse': rmse,
        'acc': acc,
        'event_loss': event_loss
    }

    note_dict = {
        'pred_dur': Pred_time_total,
        'next_dur': Next_time_total,
        'pred_type': Pred_type_total,
        'next_type': Next_type_total
    }

    if epoch % 20 == 0 or epoch == 1 :
        fig_name_prefix = "./figure/val-{val_yr}/{status}/epoch-{epoch}".format(val_yr=config.val_yr,
                                                                                status='training', epoch=epoch)
        visual(fig_name_prefix=fig_name_prefix,
               pred_time=Pred_time_total,
               gt_time=Next_time_total,
               err_dict=err_dict,
               feature_importance_inner=Fi_inner
               )


    return err_dict, note_dict


def eval_epoch(model, validation_data, pred_loss_func, epoch, tr_statistics, config):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_hits = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions

    total_event_loss = 0
    Pred_time_total, Next_time_total, Pred_type_total, Next_type_total = ([] for i in range(4))

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time, time_gap, event_type, covariates = batch


            event_time, time_gap, event_type = \
                map(lambda x: x.to(config.device), (event_time, time_gap, event_type))

            for idx, _ in enumerate(covariates):
                covariates[idx] = [inst.to(config.device) for inst in covariates[idx]]
            """ forward """
            _, prediction, _= model(event_type, event_time, covariates)

            """ compute loss """

            pred_onehot_type, pred_dur, log_prob = prediction
            # type prediction
            pred_loss, pred_num_event = Utils.type_loss(pred_onehot_type, event_type, pred_loss_func)

            # time prediction
            se = Utils.time_loss(pred_dur, time_gap)

            """ note keeping """
            # expel padding covaraites

            pred_type = torch.max(pred_onehot_type, dim=-1)[1]
            pred_type = pred_type[:, :-1].reshape(-1)
            next_type= event_type[:, 1:].reshape(-1)
            mask = next_type > 0
            Pred_type_total.append(pred_type[mask])
            Next_type_total.append(next_type[mask])
            

            next_gap = time_gap[:, 1:].reshape(-1)
            pred_dur =  pred_dur[:, :-1].reshape(-1)
            Next_time_total.append(next_gap[mask])
            Pred_time_total.append(pred_dur[mask])

            total_event_ll += log_prob.sum().item()
            total_time_se += se.item()
            total_hits += pred_num_event.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            # we do not predict the first event
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

            total_event_loss += pred_loss.item()

    Pred_time_total = [item.item() for sublist in Pred_time_total for item in sublist]
    Next_time_total = [item.item() for sublist in Next_time_total for item in sublist]
    
    # asssert cleanedList = cities[~np.isnan(Pred_time_total)]
    Pred_type_total = [item.item() for sublist in Pred_type_total for item in sublist]
    Next_type_total = [item.item() for sublist in Next_type_total for item in sublist]

    ATT = []
    for _, batch in enumerate(validation_data):
        """ prepare data """
        _, _, _, covariates = batch
        for idx, _ in enumerate(covariates):
            covariates[idx] = [inst.to(config.device) for inst in covariates[idx]]
        batch, seq_len = len(covariates), len(covariates[0])
        
        place_holder = torch.zeros(size=(batch, seq_len, model.d_covar), device=covariates[0][0].device)
        for b_id, seq in enumerate(covariates):
            # print(len(seq), type(seq), 'insight seq!')
            len_seq = len(seq)
            place_holder[b_id, :, :] = torch.cat(seq).reshape(len_seq, model.d_covar)
        place_holder = place_holder[~torch.all(place_holder==0, axis=-1)]
        att = model.encoder.cov_intra_att.get_attention(place_holder)
        ATT.append(att)

    instance_level = torch.mean(torch.cat(ATT).reshape((-1), model.d_covar), axis=0).detach().cpu().numpy()
    global_attention = model.encoder.cov_intra_att.get_mean_attention_weights().detach().cpu().numpy()





    feature_order = ['feature_{}'.format(i) for i in range(model.d_covar)]

    Fi = pd.DataFrame(columns=['feature_name', 'instance_level', 'global_attention'])
    Fi['feature_name'] = feature_order
    Fi['instance_level'] = instance_level
    Fi['global_attention'] = global_attention


    rmse = np.sqrt(total_time_se / total_num_pred)
    mean_event = total_event_ll / total_num_pred  # total_num_event
    acc = total_hits / total_num_pred 
    event_loss = total_event_loss/total_num_pred

    err_dict = {
        'log-likelihood': mean_event,
        'rmse': rmse,
        'acc': acc,
        'event_loss': event_loss
    }

    note_dict = {
        'pred_dur': Pred_time_total,
        'next_dur': Next_time_total,
        'pred_type': Pred_type_total,
        'next_type': Next_type_total
    }

    if epoch % 5 == 0 or epoch == 1:
        fig_name_prefix = "./figure/val-{val_yr}/{status}/epoch-{epoch}".format(val_yr=config.val_yr,
                                                                                status='validation', epoch=epoch)
        visual(fig_name_prefix=fig_name_prefix,
               pred_time=Pred_time_total,
               gt_time=Next_time_total,
               err_dict=err_dict,
               feature_importance_inner=Fi
               )
    return err_dict, note_dict


def train(model, training_data, validation_data, optimizer, scheduler, pred_loss_func, tr_statistics, config):
    """ Start training. """

    tr_log, val_log = ({'total_loss': [], 'log-likelihood': [], \
                        'rmse': [], 'acc': [], 'event_loss': [],\
                        'F1_': [], 'precision': [], 'recall': []} for i in range(2))
    ter_num = 0
    for epoch_i in range(config.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        # train the model
        start = time.time()
        err_dict, note_dict= train_epoch(model, training_data, optimizer, pred_loss_func, epoch, tr_statistics, config)
        print('  - (Training)    log-likelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, event_loss: {event_loss: 8.5f},'
              'elapse: {elapse:3.3f} min'
              .format(ll=err_dict['log-likelihood'], type=err_dict['acc'], rmse=err_dict['rmse'],event_loss=err_dict['event_loss'],
                      elapse=(time.time() - start) / 60))
        
        tr_log['total_loss'].append(err_dict['total_loss'].item())
        tr_log['log-likelihood'].append(err_dict['log-likelihood'])
        tr_log['event_loss'].append(err_dict['event_loss'])
        tr_log['rmse'].append(err_dict['rmse'])

        pred_type, next_type = note_dict['pred_type'], note_dict['next_type']
        next_type = np.array(next_type) -1


        F1_  = f1_score(next_type, pred_type, average='weighted')
        F1_macro  = f1_score(next_type, pred_type, average='macro')
        precision_ = precision_score(next_type, pred_type, average='weighted')
        recall_ = recall_score(next_type, pred_type, average='weighted')

        tr_log['acc'].append(err_dict['acc'])
        tr_log['acc'].append(err_dict['acc'])
        tr_log['F1_'].append(F1_)
        tr_log['precision'].append(precision_)
        tr_log['recall'].append(recall_)
        scheduler.step()
        val = True
        if val:
            # test the model
            start = time.time()
            err_dict, note_dict= eval_epoch(model, validation_data, pred_loss_func, epoch, tr_statistics, config)

            print('  - (Testing)    log-likelihood: {ll: 8.5f}, '
                'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, event_loss: {event_loss: 8.5f},'
                'elapse: {elapse:3.3f} min'
                .format(ll=err_dict['log-likelihood'], type=err_dict['acc'], rmse=err_dict['rmse'],
                        event_loss=err_dict['event_loss'],
                        elapse=(time.time() - start) / 60))

            val_log['log-likelihood'].append(err_dict['log-likelihood'])
            val_log['rmse'].append(err_dict['rmse'])
            val_log['acc'].append(err_dict['acc'])
            val_log['event_loss'].append(err_dict['event_loss'])

            pred_type, next_type = note_dict['pred_type'], note_dict['next_type']
            next_type = np.array(next_type) -1 
            
            precision_ = precision_score(next_type, pred_type, average='weighted')
            recall_ = recall_score(next_type, pred_type, average='weighted')
            val_log['F1_'].append(F1_)
            val_log['precision'].append(precision_)
            val_log['recall'].append(recall_)

            F1_  = f1_score(next_type, pred_type, average='weighted')
            F1_macro  = f1_score(next_type, pred_type, average='macro')
            print('(Testing)-Weighted-F1_score: {} and macro: {}'.format(F1_, F1_macro))


        if len(tr_log['total_loss']) >= 2:
            last_tr_err = round(tr_log['total_loss'][-2], 5)
            cur_tr_err = round(tr_log['total_loss'][-1], 5)
            rn = (last_tr_err - cur_tr_err) / last_tr_err
            if abs(rn) <= 1e-3:  # fluctating less than 0.1%
                ter_num += 1  #
            else:
                ter_num = 0 # 

            if ter_num >= 10 or epoch == config.epoch:  # reach the condition of termination: stay the same or decress less than 0.1% for successive 5 epochs.
                # tr_logging.close()

                prefix = './figure/val-{val_yr}/'.format(val_yr=config.val_yr)

                plt.figure()
                plt.plot(tr_log['log-likelihood'], label='training')
                # plt.plot(val_log['log-likelihood'], label='test')
                plt.legend()
                plt.savefig(prefix + ' log-likelihood.png')
                plt.close()

                plt.figure()
                plt.plot(tr_log['rmse'], label='training')
                # plt.plot(val_log['rmse'], label='test')
                plt.legend()
                plt.savefig(prefix + ' rmse.png')
                plt.close()

                plt.figure()
                plt.plot(tr_log['acc'], label='training')
                # plt.plot(val_log['acc'], label='test')
                plt.legend()
                plt.savefig(prefix + ' acc.png')
                plt.close()

                plt.figure()
                plt.plot(tr_log['event_loss'], label='training')
                # plt.plot(val_log['event_loss'], label='test')
                plt.legend()
                plt.savefig(prefix + ' event_loss.png')
                plt.close()

                break


def main():


    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)
    parser.add_argument('-tr_name', required=True)
    parser.add_argument('-test_name', required=True)

    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=16)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-mixture_dim', type=int, default=1)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-2)
    parser.add_argument('-smooth', type=float, default=0.1)

    parser.add_argument('-log', type=str, default='log.txt')
    parser.add_argument('-val_yr', type=int, default=2015)


    config = parser.parse_args()

    # default device is CUDA
    # config.device = torch.device('cuda')

    assert torch.cuda.is_available()

    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(config.device, 'device')
    # setup the log file
    with open(config.log, 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')
    print('[Info] parameters: {}'.format(config))

    """ prepare dataloader """
    trainloader, testloader, num_types, d_covar, tr_statistics = prepare_dataloader(config)
    d_covar = d_covar - config.delete_num # ablation study

    """ prepare model """
    model = Transformer(
        num_types=num_types,
        d_covar=d_covar,
        d_model=config.d_model,
        d_rnn=config.d_rnn,
        d_inner=config.d_inner_hid,
        n_layers=config.n_layers,
        n_head=config.n_head,
        d_k=config.d_k,
        d_v=config.d_v,
        dropout=config.dropout,
        mixture_dim=config.mixture_dim
    )
    model.to(config.device)
    with open('run.sh') as f: # save config info
        s = f.read()
        prefix = './figure/val-{val_yr}/'.format(val_yr=config.val_yr)
        with open(prefix + 'config.sh', 'w') as wf:
            wf.write(s)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                          config.lr, betas=(0.9, 0.999), eps=1e-05,weight_decay=1e-3)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if config.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(config.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, tr_statistics, config)


if __name__ == '__main__':
    main()
