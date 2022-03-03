import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils.utils import initialize_weights
import numpy as np


#*-*# for the original code


class ConcreteDropout(nn.Module):

    """Concrete Dropout.
    Implementation of the Concrete Dropout module as described in the
    'Concrete Dropout' paper: https://arxiv.org/pdf/1705.07832
    """

    def __init__(self,
                 weight_regulariser: float,
                 dropout_regulariser: float,
                 init_min: float = 0.1,
                 init_max: float = 0.1) -> None:

        """Concrete Dropout.
        Parameters
        ----------
        weight_regulariser : float
            Weight regulariser term.
        dropout_regulariser : float
            Dropout regulariser term.
        init_min : float
            Initial min value.
        init_max : float
            Initial max value.
        """

        super().__init__()

        self.weight_regulariser = weight_regulariser
        self.dropout_regulariser = dropout_regulariser

        init_min = np.log(init_min) - np.log(1.0 - init_min)
        init_max = np.log(init_max) - np.log(1.0 - init_max)

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        self.p = torch.sigmoid(self.p_logit)

        self.regularisation = 0.0

    def forward(self, x: Tensor, layer: nn.Module) -> Tensor:

        """Calculates the forward pass.
        The regularisation term for the layer is calculated and assigned to a
        class attribute - this can later be accessed to evaluate the loss.
        Parameters
        ----------
        x : Tensor
            Input to the Concrete Dropout.
        layer : nn.Module
            Layer for which to calculate the Concrete Dropout.
        Returns
        -------
        Tensor
            Output from the dropout layer.
        """

        output = layer(self._concrete_dropout(x))

        sum_of_squares = 0
        for param in layer.parameters():
            sum_of_squares += torch.sum(torch.pow(param, 2))

        weights_reg = self.weight_regulariser * sum_of_squares / (1.0 - self.p)

        dropout_reg = self.p * torch.log(self.p)
        dropout_reg += (1.0 - self.p) * torch.log(1.0 - self.p)
        dropout_reg *= self.dropout_regulariser * x[0].numel()

        self.regularisation = weights_reg + dropout_reg

        return output

    def _concrete_dropout(self, x: Tensor) -> Tensor:

        """Computes the Concrete Dropout.
        Parameters
        ----------
        x : Tensor
            Input tensor to the Concrete Dropout layer.
        Returns
        -------
        Tensor
            Outputs from Concrete Dropout.
        """

        eps = 1e-7
        tmp = 0.1

        self.p = torch.sigmoid(self.p_logit)
        u_noise = torch.rand_like(x)

        drop_prob = (torch.log(self.p + eps) -
                     torch.log(1 - self.p + eps) +
                     torch.log(u_noise + eps) -
                     torch.log(1 - u_noise + eps))

        drop_prob = torch.sigmoid(drop_prob / tmp)

        random_tensor = 1 - drop_prob
        retain_prob = 1 - self.p

        x = torch.mul(x, random_tensor) / retain_prob

        return x

def concrete_regulariser(model: nn.Module) -> nn.Module:

    """Adds ConcreteDropout regularisation functionality to a nn.Module.
    Parameters
    ----------
    model : nn.Module
        Model for which to calculate the ConcreteDropout regularisation.
    Returns
    -------
    model : nn.Module
        Model with additional functionality.
    """

    def regularisation(self) -> Tensor:

        """Calculates ConcreteDropout regularisation for each module.
        The total ConcreteDropout can be calculated by iterating through
        each module in the model and accumulating the regularisation for
        each compatible layer.
        Returns
        -------
        Tensor
            Total ConcreteDropout regularisation.
        """

        total_regularisation = 0
        for module in filter(lambda x: isinstance(x, ConcreteDropout), self.modules()):
            total_regularisation += module.regularisation

        return total_regularisation

    setattr(model, 'regularisation', regularisation)

    return model



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
        return A, x

# pMIL-V
"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class probabilistic_MIL_vanilla(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, top_k=1):
        super(probabilistic_MIL_vanilla, self).__init__()
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

        print(h.shape)
        exit()
        A, h = self.attention_net(h)

        A = torch.transpose(A, 1, 0)  # KxN

        # A_raw = A
        # A = F.softmax(A, dim=1)  # softmax over N

        dist = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(self.temperature, logits = A)
        sample = dist.rsample([16])
        asample = sample.mean(dim=0)

        M = torch.mm(asample, h)  # KxL

        # M = torch.mm(A, h) 
        logits = self.classifiers(M)
        # Y_hat = torch.topk(logits, 1, dim = 1)[1]
        # Y_prob = F.softmax(logits, dim = 1)

        # results_dict = {}

        # if return_features:
        #     results_dict.update({'features': M})
        # return logits, Y_prob, Y_hat, A_raw, results_dict

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


class probabilistic_MIL_concrete_dropout(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes=2, top_k=1):
        super(probabilistic_MIL_concrete_dropout, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        self.fc = nn.Sequential(*[nn.Linear(size[0], size[1]), nn.ReLU()])

        w, d = 1e-6, 1e-3
        self.cd1 = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d)

        if gate:
            self.attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            self.attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)

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
        self.fc = self.fc.to(device)
        self.cd1 = self.cd1.to(device)

    def forward(self, h, return_features=False):
        device = h.device
        #*-*# A, h = self.attention_net(h)  # NxK

        A = self.cd1(h, self.fc)
        A = self.attention_net(A)

        print(A.shape)
        exit()
        A = torch.transpose(A, 1, 0)  # KxN

        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, h)  # KxL

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

pMIL_model_dict = {
                    'V': probabilistic_MIL_vanilla,
                    'C': probabilistic_MIL_concrete_dropout
}

