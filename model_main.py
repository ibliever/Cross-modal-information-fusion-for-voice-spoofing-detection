"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""
import argparse
import sys
import os
import data_utils
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
import librosa

import torch
from torch import nn
from tensorboardX import SummaryWriter
import models
from models import SpectrogramModel, MFCCModel, CQCCModel, DenseNet, Bottleneck, CQCCBotNet, SpectBotNet, MFCCBotNet, LFCCBotNet, FeatureExtractor, deit_base_patch16_224, seDenseTransNet
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def pad(x, max_len=64000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = (max_len / x_len)+1
    x_repeat = np.repeat(x, num_repeats)
    padded_x = x_repeat[:max_len]
    return padded_x


def evaluate_accuracy(data_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    for batch_x, batch_y, batch_meta in data_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
    return 100 * (num_correct / num_total)


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    true_y = []
    fname_list = []
    key_list = []
    sys_id_list = []
    key_list = []
    score_list = []
    for batch_x, batch_y, batch_meta in data_loader:
        #batch_x = np.expand_dims(batch_x, axis=1)
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 1] - batch_out[:, 0]
                       ).data.cpu().numpy().ravel()

        # add outputs
        fname_list.extend(list(batch_meta[1]))
        key_list.extend(
            ['bonafide' if key == 1 else 'spoof' for key in list(batch_meta[4])])
        sys_id_list.extend([dataset.sysid_dict_inv[s.item()]
                            for s in list(batch_meta[3])])
        score_list.extend(batch_score.tolist())

    with open(save_path, 'w') as fh:
        for f, s, k, cm in zip(fname_list, sys_id_list, key_list, score_list):
            if not dataset.is_eval:
                fh.write('{} {} {} {}\n'.format(f, s, k, cm))
            else:
                fh.write('{} {} {} {}\n'.format(f, s, k, cm))
    print('Result saved to {}'.format(save_path))


def train_epoch(data_loader, model, lr, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=lr)
    weight = torch.FloatTensor([1.0, 9.0]).to(device)
    criterion = nn.NLLLoss(weight=weight)
    for batch_x, batch_y, batch_meta in train_loader:
        #batch_x = np.expand_dims(batch_x, axis=1)
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1

        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out .max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        if ii % 10 == 0:
            sys.stdout.write('\r \t {:.2f}'.format(
                (num_correct/num_total)*100))
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
    running_loss /= num_total
    train_accuracy = (num_correct/num_total)*100
    return running_loss, train_accuracy


def get_log_spectrum(x):
    s = librosa.core.stft(x, n_fft=2048, win_length=2048, hop_length=512)
    a = np.abs(s)**2
    #melspect = librosa.feature.melspectrogram(S=a)
    feat = librosa.power_to_db(a)
    return feat


def compute_mfcc_feats(x):
    mfcc = librosa.feature.mfcc(x, sr=16000, n_mfcc=24)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(delta)
    feats = np.concatenate((mfcc, delta, delta2), axis=0)
    return feats


if __name__ == '__main__':
    parser = argparse.ArgumentParser('UCLANESL ASVSpoof2019  model')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved mdoel')
    parser.add_argument('--track', type=str, default='logical')
    parser.add_argument('--features', type=str, default='cqcc')
    parser.add_argument('--is_eval', action='store_true', default=False)
    parser.add_argument('--eval_part', type=int, default=0)
    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
    track = args.track
    assert args.features in ['mfcc', 'spect', 'cqcc', 'mfcc_dense', 'spect_dense', 'cqcc_dense', 'lfcc', 'lfcc_dense', 'lfcc_sedense', 'mfcc_botnet', 'spect_botnet', 'lfcc_botnet', 'cqcc_botnet', 'lfcc_deit', 'cqcc_deit', 'mfcc_deit', 'spect_deit', 'lfcc_sedensetrans'], 'Not supported feature'
    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        track, args.features, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)
    assert track in ['logical', 'physical'], 'Invalid track given'
    is_logical = (track == 'logical')
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    if args.features == 'mfcc':
        feature_fn = compute_mfcc_feats
        model_cls = MFCCModel
        print("*************MFCCModel**********************")
    elif args.features == 'spect':
        feature_fn = get_log_spectrum
        model_cls = SpectrogramModel
        print("*************SpectrogramModel**********************")
    elif args.features == 'cqcc':
        feature_fn = None  # cqcc feature is extracted in Matlab script
        model_cls = CQCCModel
        print("*************CQCCModel**********************")
    elif args.features == 'lfcc':
        feature_fn = None  # lfcc feature is extracted in Matlab script
        model_cls = CQCCModel
        print("*************CQCCModel**********************")
    elif args.features == 'cqcc_dense':
        feature_fn = None  # cqcc feature is extracted in Matlab script
        model_cls = DenseNet
        print("*************CQCC_DenseNetModel**********************")
    elif args.features == 'spect_dense':
        feature_fn = get_log_spectrum
        model_cls = DenseNet
        print("*************spect_DneseNetModel**********************")
    elif args.features == 'mfcc_dense':
        feature_fn = compute_mfcc_feats
        model_cls = DenseNet
        print("*************mfcc_DenseNetModel**********************")    
    elif args.features == 'lfcc_dense':
        feature_fn = None
        model_cls = DenseNet
        print("*************lfcc_DenseNetModel**********************")  
    elif args.features == 'mfcc_botnet':
        feature_fn = compute_mfcc_feats
        model_cls = MFCCBotNet
        print("*************mfcc_botnet**********************")
    elif args.features == 'spect_botnet':
        feature_fn = get_log_spectrum
        model_cls = SpectBotNet
        print("*************spect_botnet**********************")
    elif args.features == 'lfcc_botnet':
        feature_fn = None
        model_cls = LFCCBotNet
        print("*************lfcc_botnet**********************")
    elif args.features == 'cqcc_botnet':
        feature_fn = None
        model_cls = CQCCBotNet
        print("*************cqcc_botnet**********************")
    elif args.features == 'lfcc_deit':
        feature_fn = None
        model_cls = models.deit_base_patch16_224
        print("*************lfcc_deit**********************")
    elif args.features == 'cqcc_deit':
        feature_fn = None
        model_cls = models.cqcc_deit_base_patch16_224
        print("*************cqcc_deit**********************")
    elif args.features == 'mfcc_deit':
        feature_fn = None
        model_cls = models.mfcc_deit_base_patch16_224
        print("*************mfcc_deit**********************")
    elif args.features == 'spect_deit':
        feature_fn = None
        model_cls = models.spect_deit_base_patch16_224
        print("*************spect_deit**********************")
    elif args.features == 'lfcc_sedensetrans':
        feature_fn = None
        model_cls = seDenseTransNet
        print("*************lfcc_seDenseTransNetModel**********************")
    transforms = transforms.Compose([
        lambda x: pad(x),
        lambda x: librosa.util.normalize(x),
        lambda x: feature_fn(x),
        lambda x: Tensor(x)
    ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dev_set = data_utils.ASVDataset(is_train=False, is_logical=is_logical,
                                    transform=transforms,
                                    feature_name=args.features, is_eval=args.is_eval, eval_part=args.eval_part)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True)
    model = model_cls().to(device)

    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    if True:
        pretrained_dict = torch.load("./models/model_logical_lfcc_dense_200_32_5e-05/epoch_200.pth")
        model_dict = model.state_dict()
        pretrained_dict_update = {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and v.size() == model_dict[k].size():
                pretrained_dict_update[k] = v
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict_update.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict_update), len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict_update)
        model.load_state_dict(model_dict)

    #model.load_state_dict(torch.load("./models/model_logical_lfcc_dense_200_32_5e-05/epoch_195.pth"))
    print(args)

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))
        print('Model loaded : {}'.format(args.model_path))

    if args.eval:
        assert args.eval_output is not None, 'You must provide an output path'
        assert args.model_path is not None, 'You must provide model checkpoint'
        produce_evaluation_file(dev_set, model, device, args.eval_output)
        sys.exit(0)

    train_set = data_utils.ASVDataset(is_train=True, is_logical=is_logical, transform=transforms,
                                      feature_name=args.features)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))

    for epoch in range(num_epochs):
        running_loss, train_accuracy = train_epoch(
            train_loader, model, args.lr, device)
        valid_accuracy = evaluate_accuracy(dev_loader, model, device)
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('valid_accuracy', valid_accuracy, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - {} - {:.2f} - {:.2f}'.format(epoch,
                                                   running_loss, train_accuracy, valid_accuracy))
        torch.save(model.state_dict(), os.path.join(
            model_save_path, 'epoch_{}.pth'.format(epoch)))
