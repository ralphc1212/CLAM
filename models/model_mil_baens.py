import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np


class dense_baens(nn.Module):
    def __init__(self, N = 4, D1 = 3, D2 = 2):
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


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()

        self.N = 4
        self.attention_a = [
            dense_baens(4, L, D),
            nn.Tanh()]

        self.attention_b = [dense_baens(4, L, D),
                            nn.Sigmoid()]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = dense_baens(4, D, n_classes)

    def forward(self, x):
        x_ = x
        x = x.unsqueeze(0)
        x = x.expand(self.N, x.shape[1], x.shape[2])
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A.sum(0), x_


class MIL_fc_baens(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, top_k=1):
        super(MIL_fc_baens, self).__init__()
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
        self.classifiers = nn.Linear(size[1], n_classes)
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

    def forward(self, h, return_features=False):
        device = h.device
        #*-*# A, h = self.attention_net(h)  # NxK        

        A, h = self.attention_net(h)

        A = torch.transpose(A, 1, 0)  # KxN

        # A = F.softmax(A, dim=1)  # softmax over N
        # M = torch.mm(A, h)

        A = F.sigmoid(A)
        M = torch.mm(A, h) / A.sum()
        # M = torch.mm(A, h)

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
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


# with prior distribution
# distance measure: MMD
# 
class MIL_fc_baens_wpr(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, top_k=1):
        super(MIL_fc_baens_wpr, self).__init__()
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
        self.classifiers = nn.Linear(size[1], n_classes)
        self.n_classes = n_classes
        self.print_sample_trigger = False
        self.num_samples = 16
        self.temperature = torch.tensor([1.0])

        self.codebook_size = 256

        self.beta = 0.25

        # self.code_book = nn.Parameter(torch.randn(codebook_size, size[1]))
        self.code_book = nn.Embedding(self.codebook_size, size[1])

        initialize_weights(self)
        self.top_k=top_k

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.temperature = self.temperature.to(device)
        self.code_book = self.code_book.to(device)

    # def prior(self):
    #     # how to encode doctor knowledge?
    #     pass

    # def codebook(self, cb_size):
    #     pass

    # def MMD(self):
    #     pass

    def vector_quantization(self, slide_embedding):

        latents_shape = slide_embedding.shape

        dist = torch.sum(slide_embedding ** 2, dim=1, keepdim=True) + \
               torch.sum(self.code_book.weight ** 2, dim=1) + \
               2 * torch.matmul(slide_embedding, self.code_book.weight.t())

        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)

        device = slide_embedding.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.codebook_size, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1 )  # [BHW x K]

        quantized_latents = torch.matmul(encoding_one_hot, self.code_book.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), slide_embedding)
        embedding_loss = F.mse_loss(quantized_latents, slide_embedding.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = slide_embedding + (quantized_latents - slide_embedding).detach()

        return vq_loss, quantized_latents

    def forward(self, h, return_features=False, slide_label=None, validation=False):
        device = h.device
        #*-*# A, h = self.attention_net(h) # NxK

        A, h = self.attention_net(h)

        A = torch.transpose(A, 1, 0)  # KxN

        # A = F.softmax(A, dim=1)  # softmax over N
        # M = torch.mm(A, h)

        A = F.sigmoid(A)
        M = torch.mm(A, h) / A.sum()

        vq_loss, M_code = self.vector_quantization(M)

        # M = torch.mm(A, h)

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

        if not validation:
            return top_instance, Y_prob, Y_hat, vq_loss, y_probs, A
        else:
            return top_instance, Y_prob, Y_hat, y_probs, A


class MIL_fc_baens_pr(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, top_k=1):
        super(MIL_fc_baens_pr, self).__init__()
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
        self.classifiers = nn.Linear(size[1], n_classes)
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

    def forward(self, h, return_features=False):
        device = h.device
        #*-*# A, h = self.attention_net(h)  # NxK        

        A, h = self.attention_net(h)

        A = torch.transpose(A, 1, 0)  # KxN

        # A = F.softmax(A, dim=1)  # softmax over N
        # M = torch.mm(A, h)

        A = F.sigmoid(A)
        M = torch.mm(A, h) / A.sum()
        # M = torch.mm(A, h)

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
        return top_instance, Y_prob, Y_hat, y_probs, results_dict
