import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from data.basic import augment_list


class Controller(nn.Module):
    def __init__(self, cfg, n_subpolicies=5, embedding_dim=32, hidden_dim=100):
        super(Controller, self).__init__()
        self.L = cfg.CONTROLLER.L
        self.T = cfg.CONTROLLER.T
        self.C = cfg.CONTROLLER.C
        self.NUM_MAGS = cfg.CONTROLLER.NUM_MAGS
        if len(cfg.CONTROLLER.EXCLUDE_OPS) > 0:
            self.NUM_OPS = len(augment_list()) - len(cfg.CONTROLLER.EXCLUDE_OPS)
        else:
            self.NUM_OPS = len(augment_list()) - cfg.CONTROLLER.EXCLUDE_OPS_NUM
        self.Q = n_subpolicies
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(self.NUM_OPS + self.NUM_MAGS, embedding_dim) # (# of operation) + (# of magnitude) 
        self.lstm = nn.LSTMCell(embedding_dim, hidden_dim)
        self.outop = nn.Linear(hidden_dim, self.NUM_OPS)
        self.outmag = nn.Linear(hidden_dim, self.NUM_MAGS)
        
        self.init_parameters()
        
    def init_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.outop.bias.data.fill_(0)
        self.outmag.bias.data.fill_(0)
        
    def get_variable(self, inputs, cuda=False, **kwargs):
        if type(inputs) in [list, np.ndarray]:
            inputs = torch.Tensor(inputs)
        if cuda:
            out = Variable(inputs.cuda(), **kwargs)
        else:
            out = Variable(inputs, **kwargs)

        return out
    
    def create_static(self, batch_size):
        inp = self.get_variable(torch.zeros(batch_size, self.embedding_dim), cuda=True, requires_grad=False)
        hx = self.get_variable(torch.zeros(batch_size, self.hidden_dim), cuda=True, requires_grad=False)
        cx = self.get_variable(torch.zeros(batch_size, self.hidden_dim), cuda=True, requires_grad=False)
        
        return inp, hx, cx
    
    def calculate(self, logits):
        probs = F.softmax(self.C * torch.tanh(logits) / self.T, dim=-1)
        log_prob = F.log_softmax(self.C * torch.tanh(logits) / self.T, dim=-1)
        entropy = -(log_prob * probs).sum(1, keepdim=False)
        action = probs.multinomial(num_samples=1).data
        selected_log_prob = log_prob.gather(1, self.get_variable(action, requires_grad=False))
        
        return entropy, probs, selected_log_prob[:, 0], action[:, 0]

    def calculate2(self, logits, action):
        log_prob = F.log_softmax(self.C * torch.tanh(logits) / self.T, dim=-1)
        selected_log_prob = log_prob.gather(1, self.get_variable(action.unsqueeze(-1), requires_grad=False))
        
        return selected_log_prob[:, 0]
    
    def forward(self, batch_size=1):
        return self.sample(batch_size)
    
    def sample(self, batch_size=1):
        policies = []
        entropies = []
        log_probs = []
        mag_probs = []
        op_probs = []
           
        for i in range(self.Q):
            inp, hx, cx = self.create_static(batch_size) # initial state
            for j in range(self.L):
                hx, cx = self.lstm(inp, (hx, cx))
                op = self.outop(hx) # B, self.NUM_OPS
                
                entropy, prob, log_prob, action = self.calculate(op)
                entropies.append(entropy)
                log_probs.append(log_prob)
                policies.append(action)
                op_probs.append(prob)
                
                # operation embedding
                inp = self.get_variable(action, requires_grad=False)
                inp = self.embedding(inp) # B, embedding_dim
                hx, cx = self.lstm(inp, (hx, cx))
                mag = self.outmag(hx) # B, self.NUM_MAGS
    
                entropy, prob, log_prob, action = self.calculate(mag)
                entropies.append(entropy)
                log_probs.append(log_prob)
                policies.append(action)
                mag_probs.append(prob)
                
                # magnitude embedding
                inp = self.get_variable(self.NUM_OPS + action, requires_grad=False) 
                inp = self.embedding(inp) # B, embedding_dim
        
        entropies = torch.stack(entropies, dim=-1) # B, Q*4
        log_probs = torch.stack(log_probs, dim=-1) # B, Q*4
        policies = torch.stack(policies, dim=-1) # B, Q*4
        op_probs = torch.stack(op_probs, dim=-1).permute(0, 2, 1).reshape(-1, self.NUM_OPS) # B, 9, Q*2
        mag_probs = torch.stack(mag_probs, dim=-1).permute(0, 2, 1).reshape(-1, self.NUM_MAGS) # B, 10, Q*2
        
        # joint probability
        return policies, torch.mean(op_probs, dim=0), torch.mean(mag_probs, dim=0), \
                torch.sum(log_probs, dim=-1), torch.sum(entropies, dim=-1) # (B, Q*4) (self.NUM_OPS,) (self.NUM_MAGS,) (B,) (B,) 

    def evaluate(self, policies, batch_size):
        log_probs = []

        for i in range(self.Q):
            inp, hx, cx = self.create_static(batch_size) # initial state
            for j in range(self.L):
                hx, cx = self.lstm(inp, (hx, cx))
                op = self.outop(hx) # B, self.NUM_OPS
                
                log_prob = self.calculate2(op, policies[:,i*self.L*2+j*2])
                log_probs.append(log_prob)
                
                # operation embedding
                inp = self.get_variable(policies[:,i*self.L*2+j*2], requires_grad=False).long()
                inp = self.embedding(inp) # B, embedding_dim
                hx, cx = self.lstm(inp, (hx, cx))
                mag = self.outmag(hx) # B, self.NUM_MAGS
    
                log_prob = self.calculate2(mag, policies[:,i*self.L*2+j*2+1])
                log_probs.append(log_prob)
                
                # magnitude embedding
                inp = self.get_variable(self.NUM_OPS + policies[:,i*self.L*2+j*2+1], requires_grad=False).long()
                inp = self.embedding(inp) # B, embedding_dim

        log_probs = torch.stack(log_probs, dim=-1) # B, Q*4

        return torch.sum(log_probs, dim=-1)
