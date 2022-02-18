import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torch import Tensor, einsum
from torch.nn import CrossEntropyLoss, BCELoss
from metrics import simplex, one_hot, one_hot2dist, class2one_hot



def search_loss(config):
    if config.CONTROLLER.LOSS == 'reinforce':
        return Reinforce(config)
    elif config.CONTROLLER.LOSS == 'ppo':
        return ProximalPolicyOptimization(config)
    else:
        raise NotImplementedError('{} is unavailable'.format(config.CONTROLLER.LOSS))


def task_loss(config):
    if config.DATASET.NAME in ['optic', 'rvs']:
        return BCELoss()
    else:
        raise NotImplementedError('Task loss is unavailable for {}'.format(config.DATASET.NAME))


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0, reduction='mean'):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target):  # pylint: disable=redefined-builtin
        log_probs = self.logsoftmax(input)
        targets = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
        if self.epsilon > 0.0:
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        targets = targets.detach()
        loss = (-targets * log_probs)
                
        if self.reduction in ['avg', 'mean']:
            loss = torch.mean(torch.sum(loss, dim=1))
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class CrossEntropy(nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropy, self).__init__()
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target):  # pylint: disable=redefined-builtin
        log_probs = self.logsoftmax(input)
        targets = target.detach()
        loss = (-targets * log_probs)
                
        if self.reduction in ['avg', 'mean']:
            loss = torch.mean(torch.sum(loss, dim=1))
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class GeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        probs = F.softmax(probs, dim=1)
        target = class2one_hot(target, 3)
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss = divided.mean()

        return loss


class Reinforce(nn.Module):
    def __init__(self, cfg):
        super(Reinforce, self).__init__()
        self.penalty = cfg.CONTROLLER.PENALTY

    def register_optimizer(self, optimizer):
        self.optimizer = optimizer

    def forward(self, controller, policies, log_probs, entropies, reward):
        score_loss = torch.mean(-log_probs * reward)
        entropy_penalty = torch.mean(entropies)
        loss = score_loss - self.penalty * entropy_penalty

        # Calculate gradients and perform backward propagation for controller
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, score_loss, entropy_penalty


class ProximalPolicyOptimization(nn.Module):
    def __init__(self, cfg):
        super(ProximalPolicyOptimization, self).__init__()
        self.clip = 0.2
        self.n_updates_per_iteration = 5
        self.penalty = cfg.CONTROLLER.PENALTY

    def register_optimizer(self, optimizer):
        self.optimizer = optimizer

    def forward(self, controller, policies, log_probs, entropies, reward):
        prev_log_probs = log_probs.detach()
        total_loss = 0
        total_score_loss = 0

        for _ in range(self.n_updates_per_iteration): 
            # Calculate V_phi and pi_theta(a_t | s_t)
            curr_log_probs = controller.evaluate(policies, reward.size(0))
            # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
            ratios = torch.exp(curr_log_probs - prev_log_probs)

            # Calculate surrogate losses.
            surr1 = ratios * reward
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * reward
            score_loss = (-torch.min(surr1, surr2)).mean()

            # entropy bonus
            entropy_penalty = torch.mean(entropies)

            loss = score_loss

            # Calculate gradients and perform backward propagation for controller
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            # accumulate loss for logging
            total_loss += loss
            total_score_loss += score_loss

        return total_loss / self.n_updates_per_iteration, total_score_loss / self.n_updates_per_iteration, entropy_penalty


class LSGAN(nn.Module):
    def __init__(self, cfg):
        super(LSGAN, self).__init__()
        self.adv = torch.nn.MSELoss()

    def forward(self, source, target):
        real_loss = self.adv(source, torch.ones_like(source))
        fake_loss = self.adv(target, torch.zeros_like(target))
        d_loss = 0.5 * (real_loss + fake_loss)

        return d_loss


class DGLSGAN(nn.Module):
    def __init__(self, cfg):
        super(DGLSGAN, self).__init__()
        self.adv = torch.nn.MSELoss()

    def forward(self, pred, gt):
        d_loss = self.adv(F.softmax(pred, dim=-1), gt)

        return d_loss


class MMD(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss