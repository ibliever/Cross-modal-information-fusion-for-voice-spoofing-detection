from torch.utils.data import DataLoader
import models
from models import FeatureExtractor, DenseNet, SEDenseNet
import data_utils
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
import librosa
import torch
from torch import nn
from collections import OrderedDict
import pickle as pkl

class IntermediateLayerGetter(nn.ModuleDict):
    """ get the output of certain layers """

    def __init__(self, model, return_layers):
        # 判断传入的return_layers是否存在于model中
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}  # 构造dict
        layers = OrderedDict()
        # 将要从model中获取信息的最后一层之前的模块全部复制下来
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)  # 将所需的网络层通过继承的方式保存下来
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        # 将所需的值以k,v的形式保存到out中
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


'''
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
    #extract_list = "fc2"
    #soft_target = torch.tensor([]).to(device)
    for batch_x, batch_y, batch_meta in data_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)
        feature = models.getFeature()
        #soft_target = torch.cat((soft_target, feature), 0)
        #print(feature)
        #extract_result = FeatureExtractor(model, extract_list)
        #batch_out = extract_result(batch_x)
        #print("batch_out is {}".format(batch_out))
        # add outputs
        fname_list = list(batch_meta[1])
        key_list = ['bonafide' if key == 1 else 'spoof' for key in list(batch_meta[4])]
        sys_id_list = [dataset.sysid_dict_inv[s.item()]
                            for s in list(batch_meta[3])]
        score_list = list(feature)

        with open(save_path, 'a+') as fh:
            for f, s, k, cm in zip(fname_list, sys_id_list, key_list, score_list):
                    fh.write('{} {} {} {}\n'.format(f, s, k, cm))
    print('Result saved to {}'.format(save_path))
    '''

def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    true_y = []
    fname_list = []
    key_list = []
    sys_id_list = []
    key_list = []
    score_list = []
    #extract_list = "fc2"
    #soft_target = torch.tensor([]).to(device)
    for batch_x, batch_y, batch_meta in data_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)
        feature = models.getFeature().cpu().detach().numpy()
        path = 'extract_pre/PA/lfcc_sedense/dev/'+batch_meta.file_name[0]+'.pkl'
        with open(path, 'ab') as f:
            pkl.dump(feature, f)


    print('Result saved to {}'.format(save_path))

def pad(x, max_len=64000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = (max_len / x_len)+1
    x_repeat = np.repeat(x, num_repeats)
    padded_x = x_repeat[:max_len]
    return padded_x

if __name__ == '__main__':
    transforms = transforms.Compose([
        lambda x: pad(x),
        lambda x: librosa.util.normalize(x),
        lambda x: Tensor(x)
    ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = data_utils.ASVDataset(is_train=False, is_logical=False,
                                    transform=transforms,
                                    feature_name='lfcc_sedense', is_eval=False, eval_part=0)

    model_cls = SEDenseNet
    model = model_cls().to(device)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.load_state_dict(torch.load('./models/model_physical_lfcc_sedense_200_32_5e-05/epoch_198.pth'))
    save_path = './extract_pre/lfcc_sedense_128.txt'
    produce_evaluation_file(dataset, model, device, save_path)
