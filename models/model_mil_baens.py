import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np

class dense_baens(nn.Module):
    def __init__(self, N=5, D1=3, D2=2):
        super(dense_baens, self).__init__()

        self.N = N
        self.D1 = D1
        self.D2 = D2

        self.U = nn.Parameter(torch.normal(0, 1, (N, D1, D2)), requires_grad=True)
        torch.nn.init.kaiming_normal_(self.U)

    def forward(self, x):

        # # baens
        # w = self.S * self.U * self.R
        # act = torch.einsum('bnd, ndl -> bnl', x, w)

        # torch.Size([8, 93829, 1024])
        # torch.Size([8, 1024, 1024])
        act = torch.einsum('ndk, nkl -> ndl', x, self.U)

        if torch.sum(torch.isnan(act)) != 0:
            print('act nan')
            print(act)
            exit()

        return act

class MIL_fc_baens(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes = 2, top_k=1):
        super(MIL_fc_baens, self).__init__()
        assert n_classes == 2
        self.size_dict = {"small": [1024, 512]}
        self.N = 8
        size = self.size_dict[size_arg]
        fc_1 = [dense_baens(N=self.N, D1=size[0], D2=size[1]), nn.ReLU()]

        if dropout:
            fc_1.append(nn.Dropout(0.25))
        self.fc_1 = nn.Sequential(*fc_1)

        self.fc_2 = dense_baens(N=self.N, D1=size[1], D2=n_classes)

        # self.sc = nn.Sequential(*[nn.Dropout(0.25)])

        # self.bn_1 = nn.BatchNorm1d(self.N)

        initialize_weights(self)
        self.top_k=top_k

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.classifier.to(device)
        self.fc_1.to(device)
        self.fc_2.to(device)
        # self.sc.to(device)

        # self.bn_1.to(device)

    def forward(self, h, return_features=False):
        h = h.unsqueeze(0).expand(self.N, -1, -1)

        # h_ = self.sc(h)

        h = self.fc_1(h)

        # h = self.bn_1((h).permute(1, 0, 2)).permute(1, 0, 2)

        # h = h + h_

        logits = self.fc_2(h).mean(dim=0)

        # if return_features:
        #     h = self.classifier.module[:3](h)
        #     logits = self.classifier.module[3](h)
        # else:
        #     logits  = self.classifier(h).mean(dim=0) # K x 1

        y_probs = F.softmax(logits, dim = 1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1) 
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_instance_idx)
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


