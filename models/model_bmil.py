import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils.utils import initialize_weights
from models.linear_vdo import LinearVDO
import numpy as np
from torch.distributions import kl

EPS_1 = 1e-16
# EPS_2 = 1e-28

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            LinearVDO(L, D, ard_init=-1.),
            nn.Tanh()]

        self.attention_b = [LinearVDO(L, D, ard_init=-1.),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = LinearVDO(D, n_classes, ard_init=-1.)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class DAttn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(DAttn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        # print(x.shape)
        return A, x

class probabilistic_MIL_Bayes(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, top_k=1):
        super(probabilistic_MIL_Bayes, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = LinearVDO(size[1], n_classes, ard_init=-3.)
        self.n_classes = n_classes
        self.print_sample_trigger = False
        self.num_samples = 16
        self.temperature = torch.tensor([1.0])

        initialize_weights(self)
        self.top_k=top_k

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.temperature = self.temperature.to(device)

    def forward(self, h, validation=False):
        device = h.device
        #*-*# A, h = self.attention_net(h)  # NxK        

        A, h = self.attention_net(h)

        A = torch.transpose(A, 1, 0)  # KxN

        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, h)
        logits = self.classifiers(M)

        y_probs = F.softmax(logits, dim = 1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1) 
        # results_dict = {}

        # if return_features:
        #     top_features = torch.index_select(h, dim=0, index=top_instance_idx)
        #     results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, A


class probabilistic_MIL_Bayes_fc(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes = 2, top_k = 1):
        super(probabilistic_MIL_Bayes_fc, self).__init__()
        assert n_classes == 2
        self.size_dict = {"small": [1024, 512]}
        size = self.size_dict[size_arg]
        fc = [LinearVDO(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))

        fc.append(LinearVDO(size[1], n_classes, ard_init=-3.))
        self.classifier = nn.Sequential(*fc)
        initialize_weights(self)
        self.top_k = top_k

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(device)

    def forward(self, h, return_features=False):
        if return_features:
            h = self.classifier.module[:3](h)
            logits = self.classifier.module[3](h)
        else:
            logits  = self.classifier(h) # K x 1

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


class probabilistic_MIL_Bayes_vis(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, top_k=1):
        super(probabilistic_MIL_Bayes_vis, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 2)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = LinearVDO(size[1], n_classes, ard_init=-3.)
        self.n_classes = n_classes
        self.print_sample_trigger = False
        self.num_samples = 16
        self.temperature = torch.tensor([1.0])

        initialize_weights(self)
        self.top_k=top_k

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.temperature = self.temperature.to(device)

    def forward(self, h, validation=False):
        device = h.device
        #*-*# A, h = self.attention_net(h)  # NxK        

        A, h = self.attention_net(h)

        # # JUST Sigmoid attn_net-n_classes = 1
        # A = torch.transpose(A, 1, 0)  # KxN
        # A = F.sigmoid(A)
        # # JUST Sigmoid

        # USING BETA attn_net-n_classes = 2
        # A = F.softplus(A, threshold=8.)
        A = F.relu(A) + EPS_1
        # print('***********************************')
        # print(A)
        # print('*max: {}, min: {}'.format(torch.max(A), torch.min(A)))

        if torch.isnan(A).sum() > 0:
            print(A)
            for k, v in self.attention_net.state_dict().items():
                print(k, v)
        postr_sp = torch.distributions.beta.Beta(A[:,0], A[:,1])
        # A = postr_sp.rsample().unsqueeze(0).clamp(min=1e-20)

        A = postr_sp.rsample().unsqueeze(0)

        print(A.shape)
        print(torch.max(A))
        exit()

        # print(torch.max(A), torch.min(A))
        # print(A)
        # print('*max: {}, min: {}'.format(torch.max(A), torch.min(A)))

        # USING DIRICHLET -> BETA attn_net-n_classes = 1
        # A = (F.relu(A) + EPS).squeeze(1)
        # # print('***********************************')
        # # print(A)
        # # print('*max: {}, min: {}'.format(torch.max(A), torch.min(A)))
        # postr_sp = torch.distributions.beta.Beta(A, A.sum() - A)
        # A = postr_sp.rsample().unsqueeze(0)
        # # print(A)
        # # print('*max: {}, min: {}'.format(torch.max(A), torch.min(A)))

        M = torch.mm(A, h)
        logits = self.classifiers(M)

        y_probs = F.softmax(logits, dim = 1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1) 
        # results_dict = {}

        # if return_features:
        #     top_features = torch.index_select(h, dim=0, index=top_instance_idx)
        #     results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, A


class probabilistic_MIL_Bayes_enc(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, top_k=1):
        super(probabilistic_MIL_Bayes_enc, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        first_transform = nn.Linear(size[0], size[1])
        fc1 = [first_transform, nn.ReLU()]
        fc2 = [first_transform, nn.ReLU()]

        if dropout:
            fc1.append(nn.Dropout(0.25))
            fc2.append(nn.Dropout(0.25))

        if gate:
            # attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
            postr_net = DAttn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
            prior_net = DAttn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
            # postr_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
            # prior_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)

        else:
            # attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
            postr_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
            prior_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)

        fc1.append(postr_net)
        fc2.append(prior_net)

        self.postr_net = nn.Sequential(*fc1)
        self.prior_net = nn.Sequential(*fc2)
        self.classifiers = LinearVDO(size[1], n_classes, ard_init=-3.)

        # self.classifiers = nn.Linear(size[1], n_classes)
        self.n_classes = n_classes
        self.print_sample_trigger = False
        self.num_samples = 16
        self.temperature = torch.tensor([1.0])
        # self.sf_pos = torch.tensor([2e4], requires_grad=False)
        # self.sf_neg = torch.tensor([2e4], requires_grad=False)
        self.sf_pos = torch.tensor([1.], requires_grad=False)
        self.sf_neg = torch.tensor([1.], requires_grad=False)
        initialize_weights(self)
        self.top_k = top_k

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.attention_net = self.attention_net.to(device)
        self.postr_net = self.postr_net.to(device)
        self.prior_net = self.prior_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.temperature = self.temperature.to(device)
        self.sf_pos = self.sf_pos.to(device)
        self.sf_neg = self.sf_neg.to(device)

    def forward(self, h, return_features=False, slide_label=None):
        device = h.device
        #*-*# A, h = self.attention_net(h)  # NxK 

        postr_alpha, h_ = self.postr_net(h)
        prior_alpha, _ = self.prior_net(h)

        # if negative, all patches should be checked with equal probabilities.
        # postr_alpha *= torch.exp(slide_label * torch.tensor([conc_expo]))

        postr_alpha = torch.transpose(postr_alpha, 1, 0)  # KxN
        prior_alpha = torch.exp(torch.transpose(prior_alpha, 1, 0))  # KxN

        # print('***************************')
        # print('before: ', postr_alpha)
        # print('component 1: ', (self.sf_pos * torch.softmax(postr_alpha / 0.1, dim=1)))
        # # print('component 1 clamp: ', (self.sf_pos * torch.softmax(postr_alpha / 0.1, dim=1)).clamp(min=1.0))
        # print('component 1 max: {}, min: {}: '.format(torch.max((self.sf_pos * torch.softmax(postr_alpha / 0.1, dim=1))),
        #     torch.min((self.sf_pos * torch.softmax(postr_alpha / 0.1, dim=1)))))
        # print('component 2: ', (self.sf_neg * torch.softmax(postr_alpha / 5., dim=1)))
        # # print('component 2 clamp: ', (self.sf_neg * torch.softmax(postr_alpha / 5., dim=1)).clamp(max=0.95))
        # print('component 2 max: {}, min: {}: '.format(torch.max((self.sf_neg * torch.softmax(postr_alpha / 5., dim=1))),
        #     torch.min((self.sf_pos * torch.softmax(postr_alpha / 5., dim=1)))))

        # postr_alpha = slide_label.detach() * (self.sf_pos * torch.softmax(postr_alpha / 0.1, dim=1)).clamp(min=1.0) \
        # + (1. - slide_label).detach() * (self.sf_neg * torch.softmax(postr_alpha / 5., dim=1)).clamp(max=0.95)

        if slide_label == 1:
            postr_alpha = (self.sf_pos * torch.softmax(postr_alpha / 0.1, dim=1)).clamp(min=1.)
        else:
            # postr_alpha = (self.sf_neg * torch.softmax(postr_alpha / 5., dim=1))
            postr_alpha = (self.sf_neg * torch.softmax(postr_alpha / 10., dim=1)).clamp(max=0.9)

        # postr_alpha = torch.exp(postr_alpha)

        # print('slide label: ', slide_label)
        # print('after: ', postr_alpha)
        # print('prior_alpha: ', prior_alpha)

        postr_kl = torch.distributions.dirichlet.Dirichlet(postr_alpha)
        postr_sp = torch.distributions.beta.Beta(postr_alpha, postr_alpha.sum() - postr_alpha)
        prior_kl = torch.distributions.dirichlet.Dirichlet(prior_alpha)
        # prior_sp = torch.distributions.beta.Beta(prior_alpha, prior_alpha.sum() - prior_alpha)

        if self.training:
            kl_div = kl.kl_divergence(postr_kl, prior_kl)
            # kl_div = kl.kl_divergence(prior_kl, postr_kl)
            A = postr_sp.rsample()
            # print('postr samples: ', A)
        else:
            prior_sp = torch.distributions.beta.Beta(prior_alpha, prior_alpha.sum() - prior_alpha)
            A = prior_sp.sample()
            # print('prior samples: ', A)

        kl_div = kl.kl_divergence(postr_kl, prior_kl)
        # kl_div = kl.kl_divergence(prior_kl, postr_kl)
        # A = 0
        # for i in range(self.num_samples):
        #     A += postr_sp.rsample()
        # A /= self.num_samples
        A = postr_sp.rsample()
        # print('postr samples: ', A)

        # print('max sample', torch.max(A))
        # print('min sample', torch.min(A))

        # A = prior_sp.rsample()

        # print('samples: ', A)
        # print('max sample', torch.max(A))
        # print('min sample', torch.min(A))

        # if positive
        # A, h = self.attention_net(h)

        # A = torch.transpose(A, 1, 0)  # KxN 

        # A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, h_)
        logits = self.classifiers(M)

        y_probs = F.softmax(logits, dim = 1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1)
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_instance_idx)
            results_dict.update({'features': top_features})
        if self.training:
            return top_instance, Y_prob, Y_hat, kl_div, y_probs, results_dict
        else:
            return top_instance, Y_prob, Y_hat, y_probs, results_dict

def get_ard_reg_vdo(module, reg=0):
    """
    :param module: model to evaluate ard regularization for
    :param reg: auxilary cumulative variable for recursion
    :return: total regularization for module
    """
    if isinstance(module, LinearVDO): return reg + module.get_reg()
    if hasattr(module, 'children'): return reg + sum([get_ard_reg_vdo(submodule) for submodule in module.children()])
    return reg


bMIL_model_dict = {
                    'A': probabilistic_MIL_Bayes,
                    'F': probabilistic_MIL_Bayes_fc,
                    'vis': probabilistic_MIL_Bayes_vis,
                    'enc': probabilistic_MIL_Bayes_enc,
}


