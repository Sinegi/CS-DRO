# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import itertools
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

#  import higher

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches
from domainbed.optimizers import get_optimizer
from domainbed import losses

from domainbed.models.resnet_mixstyle import (
    resnet18_mixstyle_L234_p0d5_a0d1,
    resnet50_mixstyle_L234_p0d5_a0d1,
)
from domainbed.models.resnet_mixstyle2 import (
    resnet18_mixstyle2_L234_p0d5_a0d1,
    resnet50_mixstyle2_L234_p0d5_a0d1,
)


def to_minibatch(x, y):
    minibatches = list(zip(x, y))
    return minibatches


class DAGWeightConstraint(object):
    def __init__(self, d, k):
        self.d = d
        self.k = k

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            # manipulate the parameters constraints
            w = w.clamp(0, None)
            for i in range(self.d):
                w[i * self.k:i * self.k + self.k, i].zero_()
            module.weight.data = w


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    transforms = {}

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

    def update(self, x, y, **kwargs):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def new_optimizer(self, parameters):
        optimizer = get_optimizer(
            self.hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = self.new_optimizer(clone.network.parameters())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class PrototypePLoss(nn.Module):
    def __init__(self, num_classes, temperature):
        super(PrototypePLoss, self).__init__()
        self.soft_plus = nn.Softplus()
        self.label = torch.arange(num_classes)
        self.temperature = temperature

    def forward(self, feature, prototypes, labels):
        feature = F.normalize(feature, p=2, dim=1)
        feature_prototype = torch.einsum('nc,mc->nm', feature, prototypes)

        feature_pairwise = torch.einsum('ic,jc->ij', feature, feature)
        mask_neg = torch.not_equal(labels, labels.T)
        l_neg = feature_pairwise * mask_neg
        l_neg = l_neg.masked_fill(l_neg < 1e-6, -np.inf)

        # [N, C+N]
        logits = torch.cat([feature_prototype, l_neg], dim=1)
        loss = F.nll_loss(F.log_softmax(logits / self.temperature, dim=1), labels)
        return loss


class MultiDomainPrototypePLoss(nn.Module):
    def __init__(self, num_classes, num_domains, temperature):
        super(MultiDomainPrototypePLoss, self).__init__()
        self.soft_plus = nn.Softplus()
        self.num_classes = num_classes
        self.label = torch.arange(num_classes)
        self.domain_label = torch.arange(num_domains)
        self.temperature = temperature

    def forward(self, feature, prototypes, labels, domain_labels):
        feature = F.normalize(feature, p=2, dim=1)
        feature_prototype = torch.einsum('nc,mc->nm', feature, prototypes.reshape(-1, prototypes.size(-1)))

        feature_pairwise = torch.einsum('ic,jc->ij', feature, feature)
        mask_neg = torch.logical_or(torch.not_equal(labels, labels.T), torch.not_equal(domain_labels, domain_labels))
        l_neg = feature_pairwise * mask_neg
        l_neg = l_neg.masked_fill(l_neg < 1e-6, -np.inf)

        # [N, C*D + N]
        logits = torch.cat([feature_prototype, l_neg], dim=1)
        loss = F.nll_loss(F.log_softmax(logits / self.temperature, dim=1), domain_labels * self.num_classes + labels)
        return loss



import numpy as np
import math


class GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer as in DANN."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x: torch.Tensor, lambd: float):
    return GradientReversal.apply(x, lambd)

class LocallyConnected(nn.Module):
    """Y_{b,j,o} = Σ_i X_{b,j,i} * W_{j,o,i} + b_{j,o}."""
    def __init__(self, d: int, in_ch: int, out_ch: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d, out_ch, in_ch))  # (j,o,i)
        self.bias = nn.Parameter(torch.empty(d, out_ch)) if bias else None
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):  # (B,d,in)
        y = torch.einsum('bji,joi->bjo', x, self.weight)
        if self.bias is not None:
            y = y + self.bias.unsqueeze(0)
        return y

############################################
# 3.  Notears-MLP DAG Module               #
############################################

class NotearsMLP(nn.Module):
    """PyTorch version of NOTEARS-MLP (Yu et al. 2019) without SciPy bounds.
    Weight constraints (non-negativity & zero-diagonal) are enforced by
    projection after each update via `.project()`.
    Args:
        dims: e.g. [d, m1, m2, …, 1] with dims[-1]==1
        bias: whether to include bias terms
    """
    def __init__(self, feat_dim: int, class_dim: int, domain_dim: int, hidden: int = 8):
        super().__init__()
        self.dims = [feat_dim+class_dim+domain_dim,hidden]
        d, m1 = self.dims[0], self.dims[1]
        self.feat_dim = feat_dim        # D
        self.class_dim = class_dim      # C
        self.domain_dim = domain_dim    # M
        self.weight_c = self.class_dim
        self.weight_d = self.class_dim * self.domain_dim
        self.d = d
        self.I = torch.eye(self.d).cuda()
        # variable-splitting first layer
        self.fc1_pos = nn.Linear(d, d * m1, bias=False)
        self.fc1_neg = nn.Linear(d, d * m1, bias=False)
        nn.init.xavier_uniform_(self.fc1_pos.weight)
        nn.init.xavier_uniform_(self.fc1_neg.weight)
        # locally connected hidden stack
        self.fc2 = nn.ModuleList()
        self.fc2.append(LocallyConnected(d, m1, 1, bias=False))
        # Augmented‑Lagrangian coeffs
        self.rho = 1.0
        self.alpha = 0.0
        self.rho_max = 1e16
        self.h_new = 1e16
        self.K = int(math.sqrt(self.feat_dim))
        self.register_projection_mask()
    
    def register_projection_mask(self):
        D, C, M = self.feat_dim, self.class_dim, self.domain_dim
        d, m1 = self.dims[0], self.dims[1]
        
        # 전체 shape: (d, m1, d)
        mask = torch.ones((d, m1, d), device=self.fc1_pos.weight.device)
        
        idx = torch.arange(d, device=mask.device)
        mask[idx, :, idx] = 0  # zero diagonal

        feat_idx   = torch.arange(0, D, device=mask.device)
        class_idx  = torch.arange(D, D+C, device=mask.device)
        domain_idx = torch.arange(D+C, D+C+M, device=mask.device)

        # block class ↔ class
        mask[class_idx[:, None], :, class_idx] = 0
        # block domain ↔ domain
        mask[domain_idx[:, None], :, domain_idx] = 0
        # block class ↔ domain (both directions)
        mask[class_idx[:, None], :, domain_idx] = 0
        mask[domain_idx[:, None], :, class_idx] = 0
        # block class → feature
        mask[feat_idx[:, None], :, class_idx] = 0
        # block domain → feature
        mask[feat_idx[:, None], :, domain_idx] = 0

        self.register_buffer("projection_mask", mask)  # no grad

    # ------------------ forward ------------------
    def forward_(self, xx: torch.Tensor):  # (B,d) -> (B,d)
        self.project()
        d, m1 = self.dims[0], self.dims[1]
        B,D = xx.shape
        x = self.fc1_pos(xx) - self.fc1_neg(xx)        # (B,d*m1)
        x = x.view(-1, d, m1)                        # (B,d,m1)
        for fc in self.fc2:
            x = torch.sigmoid(x)
            x = fc(x)                                # maintain shape (B,d,*)
        x = x.squeeze(-1)                         # final dims[-1]==1 → (B,d)
        return x
    
    def get_adjacency_matrix(self):
        """
        Computes soft adjacency matrix A from fc1 weights.
        Returns A: torch.Tensor of shape [d, d]
        """
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # shape: [d * h, d]
        fc1_weight = fc1_weight.view(d, -1, d)     # shape: [d, h, d]
        A = torch.sum(fc1_weight ** 2, dim=1).t()               # shape: [d, d]
        return A

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        A = self.get_adjacency_matrix()
        # h = self.h_truncated(A,K=self.K)
        d = A.size(0)
        M = torch.eye(d,device=A.device) + A / d  # (Yu et al. 2019)
        E = torch.matrix_power(M, d - 1)
        h = torch.trace(E.t() * M).sum() - d
        return h

    def l2_reg(self):
        reg = (self.fc1_pos.weight - self.fc1_neg.weight).pow(2).sum()
        for fc in self.fc2:
            reg += fc.weight.pow(2).sum()
        return reg

    def l1_reg(self):
        return self.fc1_pos.weight.sum() + self.fc1_neg.weight.sum()

    @torch.no_grad()
    def project(self):
        self.fc1_pos.weight.data.clamp_(0)
        self.fc1_neg.weight.data.clamp_(0)

        d, m1 = self.dims[0], self.dims[1]
        w_pos = self.fc1_pos.weight.data.view(d, m1, d)
        w_neg = self.fc1_neg.weight.data.view(d, m1, d)

        w_pos.mul_(self.projection_mask)
        w_neg.mul_(self.projection_mask)
        
    
    def get_mask(self, gamma=20.0, quantile=0.5):
        """Get mask for the first layer weights."""
        A = self.get_adjacency_matrix().t()  # (D+C+M, D+C+M)
        A_class = A[self.feat_dim:self.feat_dim+self.class_dim, :self.feat_dim]  # (C,D)
        score_class  = A_class.sum(dim=0)  # (D,)
        A_domain = A[self.feat_dim+self.class_dim:, :self.feat_dim]         # shape [M, D]
        score_domain = A_domain.sum(dim=0)                    # shape [D]
        # if self.training:
        #     print("class:", score_class.mean().item(), score_class.max().item(), score_class.min().item())
        #     print("domain:", score_domain.mean().item(), score_domain.max().item(), score_domain.min().item())
        #     aaa = score_class - score_domain   
        #     print("diff:", aaa.mean().item(), aaa.max().item(), aaa.min().item())
        score_class =(score_class-score_class.min())/(score_class.max()-score_class.min()).clamp(min=1e-6)
        score_domain =(score_domain-score_domain.min())/(score_domain.max()-score_domain.min()).clamp(min=1e-6)
        score_diff = score_class - score_domain                # shape [D]
        score_diff =(score_diff-score_diff.min())/(score_diff.max()-score_diff.min()).clamp(min=1e-6)
        threshold = torch.quantile(score_diff, quantile)
        shifted = score_diff - threshold
        # 🔹 Sigmoid masking
        mask = torch.sigmoid(gamma*shifted)  # (D,)
        return mask
    
    def forward_reverse(self, x: torch.Tensor, x_: torch.Tensor):
        """Forward pass with mask applied to the first layer weights."""
        self.project()
        mask = self.get_mask()   # (D)
        x_prime = x + (x_-x) * (mask.view(1, -1))
        scale = x.norm(dim=-1,p=2,keepdim=True) / x_prime.norm(dim=-1,p=2,keepdim=True).clamp(min=1e-6)
        x_prime = x_prime * scale
        return x_prime
    
    def forward(self, x: torch.Tensor, x_: torch.Tensor):
        """Forward pass with mask applied to the first layer weights."""
        self.project()
        mask = self.get_mask()   # (D)
        x_prime = x + (x_-x) * (1-mask.view(1, -1))
        scale = x.norm(dim=-1,p=2,keepdim=True) / x_prime.norm(dim=-1,p=2,keepdim=True).clamp(min=1e-6)
        x_prime = x_prime * scale
        return x_prime
    
    # ---------- likelihood loss ------
    def dag_loss(self, feature: torch.Tensor, 
                 step: int, interval: int = 100, lambda_reg: float = 1e-4):
        B, D = feature.shape # (B,V,D+C+M)
        # print(step)
        # μ_hat
        recon = self.forward_(feature)           # (B,V,D+C+M)
        loss = F.mse_loss(recon[:,:self.feat_dim], feature[:,:self.feat_dim], reduction='sum')  # MSE loss
        loss += self.weight_c*F.mse_loss(recon[:,self.feat_dim:self.feat_dim+self.class_dim], feature[:,self.feat_dim:self.feat_dim+self.class_dim], reduction='sum')  # MSE loss
        loss += self.weight_d*F.mse_loss(recon[:,self.feat_dim+self.class_dim:], feature[:,self.feat_dim+self.class_dim:], reduction='sum')  # MSE loss
        loss /= B
        # acyclicity penalty
        alc = 0
        if step > interval:
            h_val = self.h_func()
            if h_val.item() < 1e-8:
                return 0, False
            if step % interval == 0:
                if self.rho < self.rho_max:
                    if h_val.item() > 0.25 * self.h_new:
                        self.rho *= 10
                    else:
                        self.alpha += self.rho * h_val.item()
                        self.h_new = h_val.item()
            alc = 0.5 * self.rho * h_val.pow(2) + self.alpha * h_val
        reg = self.l2_reg() + self.l1_reg()
        return loss + lambda_reg * reg + alc, True

class DomainClassifier(nn.Module):
    def __init__(self, feat_dim: int, num_domains: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, num_domains),
        )
    def forward(self, x: torch.Tensor):
        return self.classifier(x)


class iDAG(Algorithm):
    """
    DAG domain generalization methods

    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(iDAG, self).__init__(input_shape,
                                   num_classes,
                                   num_domains,
                                   hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.encoder = networks.LightEncoder(self.featurizer.n_outputs,
                                             hparams['out_dim'],
                                             hparams['hidden_size'],
                                             hparams["num_hidden_layers"])
        self.f_dim = hparams['out_dim']
        
        self.classifier = nn.Linear(self.f_dim, num_classes)
        
        self.num_classes = num_classes
        self.num_domains = num_domains
        
        
        self.l_ica = 0.01
        self.l_dom = 1.0
        self.grl_lambda = 1.0
        self.size_dataloader = 200
        
        
        
        self.domain_clf = DomainClassifier(self.f_dim, num_domains)
        
        self.proto_momentum = 0.99
        self.register_buffer("prototypes_z", torch.zeros(num_classes, self.f_dim))
        self.register_buffer("prototypes_c", torch.zeros(num_classes,num_classes))
        self.register_buffer("prototypes_d", torch.zeros(num_domains,num_domains))
        
        self.dag_module = NotearsMLP(self.f_dim,num_classes,num_domains)
        
        self.network = nn.Sequential(self.featurizer, self.encoder, self.classifier)
        
        # optimizer
        parameters = [
            {"params": self.featurizer.parameters()},
            {"params": self.encoder.parameters()},
            {"params": self.classifier.parameters()},
            {"params": self.domain_clf.parameters()},
        ]
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        
        parameters_dag = [
            {"params": self.dag_module.parameters()},
        ]
        self.optimizer_dag = get_optimizer(
            hparams["optimizer"],
            parameters_dag,
            lr=1e-4,#self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
    
    def update_prototypes(self, features, labels):
        """
        features: [B, V, D]
        labels: [B]
        """
        with torch.no_grad():
            for c in range(self.num_classes):
                mask = labels == c  # 해당 클래스의 마스크
                num_c = mask.sum().item()
                if num_c == 0:
                    continue  # 해당 클래스 샘플 없으면 스킵
                f_c = features[mask]  # (Nc, V, D)
                mean_feat = f_c.mean(dim=0)  # (V, D)
                proto = self.prototypes_z[c]
                if proto.norm().item() == 0:
                    self.prototypes_z[c] = mean_feat.detach()
                else:
                    self.prototypes_z[c] = (
                        proto * self.proto_momentum + (1 - self.proto_momentum) * mean_feat.detach()
                    )
                        
    def update_probabilities(self, c_probs, d_probs, y, dom):
        with torch.no_grad():
            for c in range(self.num_classes):
                mask = y == c
                if mask.sum() ==0:
                    continue
                mean_c = c_probs[mask].mean(dim=0)
                if self.prototypes_c[c].norm().item() == 0:
                    self.prototypes_c[c] = mean_c.detach()
                else:
                    self.prototypes_c[c] = self.proto_momentum * self.prototypes_c[c] + (1 - self.proto_momentum) * mean_c
            for m in range(self.num_domains):
                mask = dom == m
                if mask.sum() ==0:
                    continue
                mean_d = d_probs[mask].mean(dim=0)
                if self.prototypes_d[m].norm().item() == 0:
                    self.prototypes_d[m] = mean_d.detach()
                else:
                    self.prototypes_d[m] = self.proto_momentum * self.prototypes_d[m] + (1 - self.proto_momentum) * mean_d
    
    def ica_relu_aware_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        ICA-inspired decorrelation loss tailored for ReLU outputs.
        입력:
            z: [B, D] (ReLU를 거친 임베딩)
        출력:
            scalar loss tensor
        """
        B, D = z.shape
        eps = 1e-8

        # Normalize (zero mean, unit variance)
        z_norm = (z - z.mean(dim=0, keepdim=True)) / (z.std(dim=0, keepdim=True) + eps)

        # 1. Linear decorrelation
        cov_lin = (z_norm.T @ z_norm) / B
        off_diag_lin = cov_lin - torch.diag(torch.diag(cov_lin))

        # 2. Nonlinear decorrelation using log(1 + z)
        z_nl = torch.tanh(z)  # log(1 + z), safe for ReLU outputs (z >= 0)
        z_nl = (z_nl - z_nl.mean(dim=0, keepdim=True)) / (z_nl.std(dim=0, keepdim=True) + eps)
        cov_nl = (z_nl.T @ z_nl) / B
        off_diag_nl = cov_nl - torch.diag(torch.diag(cov_nl))

        # 3. Combine both terms
        loss = (off_diag_lin ** 2).mean() + (off_diag_nl ** 2).mean()
        return loss
    
    def _cross_entropy(self, logits, y):
        loss = F.cross_entropy(logits, y)
        prob = logits.softmax(dim=-1)
        ent  = (-prob * (prob + 1e-8).log()).sum(-1).mean()      # H(p)
        max_ent = math.log(prob.size(-1))
        confusion = (max_ent - ent)                     # 작아질수록 uniform
        return loss + 0.01*confusion

    def update(self, x, y, **kwargs):
        dom = []
        for i in range(len(x)):
            dom.append(torch.ones_like(y[i],device=y[i].device) * i)
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        all_dom = torch.cat(dom)
        
        all_f = self.featurizer(all_x)
        feat = self.encoder(all_f)
        logit = self.classifier(feat)
        loss = self._cross_entropy(logit, all_y)
        
        # dom_logit = self.domain_clf(grad_reverse(feat, self.grl_lambda))
        dom_logit = self.domain_clf(feat.detach())
        dom_loss = self._cross_entropy(dom_logit, all_dom)
        ica_loss = self.ica_relu_aware_loss(feat)
        reg_loss = self.l_ica*ica_loss + self.l_dom*dom_loss 
        
        loss_interv = torch.tensor(0.0, device=all_x.device)
        if kwargs["step"] > self.size_dataloader//2:
            self.update_probabilities(logit, dom_logit, all_y, all_dom)
            self.update_prototypes(feat, all_y)
        if kwargs["step"] > 2*self.size_dataloader:
            input_dag = torch.cat([feat, logit, dom_logit], dim=-1).detach()
            out_recon =self.dag_module.forward_(input_dag)
            feat_recon = out_recon[:,:feat.shape[-1]]  # (B, D)
            z_causal  = self.dag_module(feat, feat_recon) # causal part 강조 → classifier용
            logit_causal = self.classifier(z_causal )
            loss_interv += self._cross_entropy(logit_causal, all_y)
            
            z_spurious  = self.dag_module.forward_reverse(feat, feat_recon) # spurious part 강조 → domain classifier용
            logit_spurious = self.domain_clf(grad_reverse(z_spurious, self.grl_lambda))
            loss_interv += 0.1*self._cross_entropy(logit_spurious, all_dom)
    
        loss = loss + reg_loss + 0.5*loss_interv
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
        # DAG loss
        if kwargs["step"] > self.size_dataloader :
            with torch.no_grad():
                f_data = self.prototypes_z[all_y]       # (B,D)
                c_data = self.prototypes_c[all_y]       # [B,C]
                d_data = self.prototypes_d[all_dom]     # [B,M]
                # input_dag = F.normalize(torch.cat([f_data, c_data, d_data], dim=-1).detach(),dim=-1)
                input_dag = torch.cat([f_data, c_data, d_data], dim=-1).detach()
            dag_loss, flag = self.dag_module.dag_loss(input_dag, kwargs["step"], interval=self.size_dataloader)
            if flag:
                self.optimizer_dag.zero_grad()
                dag_loss.backward()
                self.optimizer_dag.step()
            else:
                dag_loss = torch.tensor(0.0, device=all_x.device)
        else:
            dag_loss = torch.tensor(0.0, device=all_x.device)

        return {"loss": loss.item(), "reg_loss": reg_loss.item(), "loss_interv":loss_interv.item(), "dag_loss": dag_loss.item()}

    def predict(self, x):
        feat = self.encoder(self.featurizer(x))
        logit = self.classifier(feat)
        return logit

    def clone(self):
        clone = copy.deepcopy(self)
        params = [
            {"params": clone.network.parameters()},
        ]
        clone.optimizer = self.new_optimizer(params)
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone


class iDAGamp(iDAG):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(iDAGamp, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.accumulation_steps = 2
    
    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        domain_labels = torch.cat([torch.ones(len(_y)) * i for i, _y in enumerate(y)]).long().to(all_x.device)

        all_f = self.featurizer(all_x)
        all_f = self.encoder(all_f)
        all_masked_f = self.dag_mlp(all_f)

        for f, masked_f, label_y, label_d in zip(F.normalize(all_f, dim=1), 
                                                 F.normalize(all_masked_f, dim=1), 
                                                 all_y, 
                                                 domain_labels):
            self.prototypes[label_d, label_y] = self.prototypes[label_d, label_y] * self.proto_m + (1 - self.proto_m) * f.detach()
            self.prototypes_y[label_y] = self.prototypes_y[label_y] * self.proto_m + (1 - self.proto_m) * masked_f.detach()
        self.prototypes = F.normalize(self.prototypes, p=2, dim=2)
        self.prototypes_y = F.normalize(self.prototypes_y, p=2, dim=1)

        prototypes = self.prototypes.detach().clone()
        prototypes_y = self.prototypes_y.detach().clone()

        proto_rec, masked_proto = self.dag_mlp(
            x=prototypes.view(self.num_domains * self.num_classes, -1),
            y=self.prototypes_label)

        # reconstruction loss
        loss_rec = F.cosine_embedding_loss(
            proto_rec, 
            prototypes.view(self.num_domains * self.num_classes, -1),
            torch.ones(self.num_domains * self.num_classes, device=all_x.device))
        loss_rec += F.cross_entropy(
            self.rec_classifier(masked_proto),
            self.prototypes_label)
        loss_rec = self.lambda2 * loss_rec
        h_val = self.dag_mlp.h_func()
        penalty = 0.5 * self.rho * h_val * h_val + self.alpha * h_val
        l1_reg = self.lambda1 * self.dag_mlp.w_l1_reg()

        # update the DAG hyper-parameters
        if kwargs['step'] % 100 == 0:
            if self.rho < self.rho_max and h_val > 0.25 * self._h_val:
                self.rho *= 10
                self.alpha += self.rho * h_val.item()
            self._h_val = h_val.item()

        loss_dag = loss_rec + penalty + l1_reg

        loss_inv_ce = F.cross_entropy(self.inv_classifier(all_masked_f), all_y)

        loss_contr_mu = self.hparams["weight_mu"] * self.loss_proto_con(all_masked_f, prototypes_y, all_y)
        loss_contr_nu = self.hparams["weight_nu"] * self.loss_multi_proto_con(all_f, prototypes, all_y, domain_labels)
        loss_contr = loss_contr_mu + loss_contr_nu

        if kwargs['step'] == self.hparams["dag_anneal_steps"]:
            # avoid the gradient jump
            params = [
                {"params": self.network.parameters()},
            ]
            self.optimizer = get_optimizer(
                self.hparams["optimizer"],
                params,
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        if kwargs['step'] >= self.hparams["dag_anneal_steps"]:
            loss = loss_inv_ce + loss_dag + loss_contr_mu + loss_contr_nu
        else:
            loss = loss_inv_ce + loss_contr_nu

        loss = loss / self.accumulation_steps
        loss.backward()
        if (kwargs['step'] + 1) % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        # constraint DAG weights
        self.dag_mlp.projection()

        return {"loss": loss.item(),
                "inv_ce": loss_inv_ce.item(),
                "l2": loss_rec.item(),
                "penalty": penalty.item(),
                "l1": l1_reg.item(),
                "cl": loss_contr.item()}


class iDAGCMNIST(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(iDAGCMNIST, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.notears_mlp = networks.LinearNotears(self.featurizer.n_outputs + 1)
        self.notears_optimizer = torch.optim.Adam(self.notears_mlp.parameters(),
                                                  lr=self.hparams["lr"])
        self.lambda1 = hparams['lambda1']
        self.lambda2 = hparams['lambda2']
        self.h_tol = hparams['h_tol']
        self.rho_max = hparams['rho_max']
        self.rho = 1e8
        self.alpha = 1e8
        self.h = np.inf
        self.train_dag = False

        # create prototype
        self.proto_m = 0.99
        self.register_buffer("prototypes", torch.zeros(num_classes, self.featurizer.n_outputs))
        self.register_buffer("prototypes_label", torch.arange(num_classes).unsqueeze(1))
        self.loss_proto_con = PrototypePLoss(num_classes, 0.07)

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        domain_labels = torch.cat(
            [torch.ones(len(_y)) * i for i, _y in enumerate(y)]).long().to(all_x.device)
        all_f = self.featurizer(all_x)

        for f, label_y in zip(all_f, all_y):
            self.prototypes[label_y] = self.proto_m * self.prototypes[label_y] + (1 - self.proto_m) * f.detach()
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

        prototypes = self.prototypes.clone().detach()
        loss_contr = self.loss_proto_con(all_f, prototypes, all_y)

        # dag_inputs = torch.cat([all_f, all_y.unsqueeze(1)], dim=1)
        dag_inputs = torch.cat([prototypes, self.prototypes_label], dim=1)
        dag_inputs = dag_inputs - dag_inputs.mean(dim=0, keepdim=True)

        notears_pos = self.notears_mlp(dag_inputs)
        loss_ce = F.binary_cross_entropy_with_logits(notears_pos[:, -1], self.prototypes_label.squeeze().float())

        # constraint weights
        self.notears_mlp.weight_pos.data.clamp_(0, None)
        self.notears_mlp.weight_neg.data.clamp_(0, None)
        self.notears_mlp.weight_pos.data.fill_diagonal_(0)
        self.notears_mlp.weight_neg.data.fill_diagonal_(0)

        h_val = self.notears_mlp.h_func()
        penalty = 0.5 * self.rho * h_val * h_val + self.alpha * h_val
        l1_reg = self.lambda1 * self.notears_mlp.w_l1_reg()

        if kwargs["step"] % 50 == 0:
            self.train_dag = ~self.train_dag
            if self.h > 1.0:
                self.train_dag = True
                self.notears_optimizer = torch.optim.Adam(
                    self.notears_mlp.parameters(),
                    lr=self.hparams["lr"])

        loss_rec = F.mse_loss(notears_pos[:, :-1],
                              dag_inputs[:, :-1].detach(),
                              reduction='sum') / all_f.size(0) * 0.5

        loss_dag = loss_rec + penalty + l1_reg
        if kwargs["step"] < self.hparams["warmup_steps"]:
            # Warmup phase
            loss = loss_ce + loss_rec + loss_contr
        elif self.train_dag:
            # train dag
            for p in self.featurizer.parameters():
                p.requires_grad = False
            for p in self.notears_mlp.parameters():
                p.requires_grad = True
            loss = loss_ce + loss_dag
        else:
            # train featurizer
            for p in self.featurizer.parameters():
                p.requires_grad = True
            for p in self.notears_mlp.parameters():
                p.requires_grad = False
            loss = loss_ce + loss_rec + loss_contr

        self.h = h_val.item()

        self.notears_optimizer.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.notears_optimizer.step()
        self.optimizer.step()

        return {"loss": loss.item(), "loss_ce": loss_ce.item(), "loss_rec": loss_rec.item(),
                "penalty": penalty.item(), "l1_reg": l1_reg.item(), "loss_contr": loss_contr.item()}

    def predict(self, x):
        f = self.featurizer(x)
        dag_inputs = torch.cat([f, torch.zeros(f.size(0), 1, device=f.device)], dim=1)
        dag_inputs = dag_inputs - dag_inputs.mean(dim=0, keepdim=True)
        notears_f = self.notears_mlp(dag_inputs)
        logits = torch.sigmoid(notears_f[:, -1:])
        logits = torch.cat([1.0 - logits, logits], dim=1)
        return logits


class Mixstyle(Algorithm):
    """MixStyle w/o domain label (random shuffle)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
        super().__init__(input_shape, num_classes, num_domains, hparams)
        if hparams["resnet18"]:
            network = resnet18_mixstyle_L234_p0d5_a0d1()
        else:
            network = resnet50_mixstyle_L234_p0d5_a0d1()
        self.featurizer = networks.ResNet(input_shape, self.hparams, network)

        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class Mixstyle2(Algorithm):
    """MixStyle w/ domain label"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
        super().__init__(input_shape, num_classes, num_domains, hparams)
        if hparams["resnet18"]:
            network = resnet18_mixstyle2_L234_p0d5_a0d1()
        else:
            network = resnet50_mixstyle2_L234_p0d5_a0d1()
        self.featurizer = networks.ResNet(input_shape, self.hparams, network)

        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def pair_batches(self, xs, ys):
        xs = [x.chunk(2) for x in xs]
        ys = [y.chunk(2) for y in ys]
        N = len(xs)
        pairs = []
        for i in range(N):
            j = i + 1 if i < (N - 1) else 0
            xi, yi = xs[i][0], ys[i][0]
            xj, yj = xs[j][1], ys[j][1]

            pairs.append(((xi, yi), (xj, yj)))

        return pairs

    def update(self, x, y, **kwargs):
        pairs = self.pair_batches(x, y)
        loss = 0.0

        for (xi, yi), (xj, yj) in pairs:
            #  Mixstyle2:
            #  For the input x, the first half comes from one domain,
            #  while the second half comes from the other domain.
            x2 = torch.cat([xi, xj])
            y2 = torch.cat([yi, yj])
            loss += F.cross_entropy(self.predict(x2), y2)

        loss /= len(pairs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class ARM(ERM):
    """Adaptive Risk Minimization (ARM)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams["batch_size"]

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class SAM(ERM):
    """Sharpness-Aware Minimization
    """
    @staticmethod
    def norm(tensor_list: List[torch.tensor], p=2):
        """Compute p-norm for tensor list"""
        return torch.cat([x.flatten() for x in tensor_list]).norm(p)

    def update(self, x, y, **kwargs):
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        # 1. eps(w) = rho * g(w) / g(w).norm(2)
        #           = (rho / g(w).norm(2)) * g(w)
        grad_w = autograd.grad(loss, self.network.parameters())
        scale = self.hparams["rho"] / self.norm(grad_w)
        eps = [g * scale for g in grad_w]

        # 2. w' = w + eps(w)
        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.add_(v)

        # 3. w = w - lr * g(w')
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        # restore original network params
        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.sub_(v)
        self.optimizer.step()

        return {"loss": loss.item()}


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.register_buffer("update_count", torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.discriminator = networks.MLP(self.featurizer.n_outputs, num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes, self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = get_optimizer(
            hparams["optimizer"],
            (list(self.discriminator.parameters()) + list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams["weight_decay_d"],
            betas=(self.hparams["beta1"], 0.9),
        )

        self.gen_opt = get_optimizer(
            hparams["optimizer"],
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams["weight_decay_g"],
            betas=(self.hparams["beta1"], 0.9),
        )

    def update(self, x, y, **kwargs):
        self.update_count += 1
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])
        minibatches = to_minibatch(x, y)
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat(
            [
                torch.full((x.shape[0],), i, dtype=torch.int64, device="cuda")
                for i, (x, y) in enumerate(minibatches)
            ]
        )

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1.0 / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction="none")
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(
            disc_softmax[:, disc_labels].sum(), [disc_input], create_graph=True
        )[0]
        grad_penalty = (input_grad ** 2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams["grad_penalty"] * grad_penalty

        d_steps_per_g = self.hparams["d_steps_per_g_step"]
        if self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g:

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {"disc_loss": disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = classifier_loss + (self.hparams["lambda"] * -disc_loss)
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {"gen_loss": gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(AbstractDANN):
    """Unconditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=False,
            class_balance=False,
        )


class CDANN(AbstractDANN):
    """Conditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=True,
            class_balance=True,
        )


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1.0).cuda().requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        penalty_weight = (
            self.hparams["irm_lambda"]
            if self.update_count >= self.hparams["irm_penalty_anneal_iters"]
            else 1.0
        )
        nll = 0.0
        penalty = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams["irm_penalty_anneal_iters"]:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = get_optimizer(
                self.hparams["optimizer"],
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams["vrex_penalty_anneal_iters"]:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = get_optimizer(
                self.hparams["optimizer"],
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains, hparams)

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class OrgMixup(ERM):
    """
    Original Mixup independent with domains
    """

    def update(self, x, y, **kwargs):
        x = torch.cat(x)
        y = torch.cat(y)

        indices = torch.randperm(x.size(0))
        x2 = x[indices]
        y2 = y[indices]

        lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

        x = lam * x + (1 - lam) * x2
        predictions = self.predict(x)

        objective = lam * F.cross_entropy(predictions, y)
        objective += (1 - lam) * F.cross_entropy(predictions, y2)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class CutMix(ERM):
    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def update(self, x, y, **kwargs):
        # cutmix_prob is set to 1.0 for ImageNet and 0.5 for CIFAR100 in the original paper.
        x = torch.cat(x)
        y = torch.cat(y)

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(x.size()[0]).cuda()
            target_a = y
            target_b = y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
            # compute output
            output = self.predict(x)
            objective = F.cross_entropy(output, target_a) * lam + F.cross_entropy(
                output, target_b
            ) * (1.0 - lam)
        else:
            output = self.predict(x)
            objective = F.cross_entropy(output, y)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q) / len(minibatches)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains, hparams)

    def update(self, x, y, **kwargs):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        minibatches = to_minibatch(x, y)
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = get_optimizer(
                self.hparams["optimizer"],
                #  "SGD",
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # 1. Compute supervised loss for meta-train set
            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(), inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # 2. Compute meta loss for meta-val set
            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(), allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams["mldg_beta"] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(self.hparams["mldg_beta"] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {"loss": objective}


#  class SOMLDG(MLDG):
#      """Second-order MLDG"""
#      # This commented "update" method back-propagates through the gradients of
#      # the inner update, as suggested in the original MAML paper.  However, this
#      # is twice as expensive as the uncommented "update" method, which does not
#      # compute second-order derivatives, implementing the First-Order MAML
#      # method (FOMAML) described in the original MAML paper.

#      def update(self, x, y, **kwargs):
#          minibatches = to_minibatch(x, y)
#          objective = 0
#          beta = self.hparams["mldg_beta"]
#          inner_iterations = self.hparams.get("inner_iterations", 1)

#          self.optimizer.zero_grad()

#          with higher.innerloop_ctx(
#              self.network, self.optimizer, copy_initial_weights=False
#          ) as (inner_network, inner_optimizer):
#              for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
#                  for inner_iteration in range(inner_iterations):
#                      li = F.cross_entropy(inner_network(xi), yi)
#                      inner_optimizer.step(li)

#                  objective += F.cross_entropy(self.network(xi), yi)
#                  objective += beta * F.cross_entropy(inner_network(xj), yj)

#              objective /= len(minibatches)
#              objective.backward()

#          self.optimizer.step()

#          return {"loss": objective.item()}


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains, hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(
            x1_norm
        )
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=(0.001, 0.01, 0.1, 1, 10, 100, 1000)):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= nmb * (nmb - 1) / 2

        self.optimizer.zero_grad()
        (objective + (self.hparams["mmd_gamma"] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {"loss": objective.item(), "penalty": penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=False)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs * 2, num_classes)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.register_buffer("embeddings", torch.zeros(num_domains, self.featurizer.n_outputs))

        self.ema = self.hparams["mtl_ema"]

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding + (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))


class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains, hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = nn.Linear(self.network_f.n_outputs, num_classes)
        # style network
        self.network_s = nn.Linear(self.network_f.n_outputs, num_classes)

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return get_optimizer(
                hparams["optimizer"], p, lr=hparams["lr"], weight_decay=hparams["weight_decay"]
            )

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).cuda()

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, x, y, **kwargs):
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {
            "loss_c": loss_c.item(),
            "loss_s": loss_s.item(),
            "loss_adv": loss_adv.item(),
        }

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.drop_f = (1 - hparams["rsc_f_drop_factor"]) * 100
        self.drop_b = (1 - hparams["rsc_b_drop_factor"]) * 100
        self.num_classes = num_classes

    def update(self, x, y, **kwargs):
        # inputs
        all_x = torch.cat([xi for xi in x])
        # labels
        all_y = torch.cat([yi for yi in y])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.cuda()).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Perform all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.

    :param tensor: Input tensot.
    :type tensor: torch.Tensor
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

@torch.no_grad()
def average_all_gather(tensor):
    """
    Perform all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.

    :param tensor: Input tensot.
    :type tensor: torch.Tensor
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.mean(torch.stack(tensors_gather), dim=0)
    return output
