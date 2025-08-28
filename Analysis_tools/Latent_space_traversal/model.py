import numpy as np
import torch
from utils import *

class MuRP(torch.nn.Module):
    def __init__(self, d, dim, device):
        super(MuRP, self).__init__()
        self.device = device
        
        self.Eh = torch.nn.Embedding(len(d.entities), dim, padding_idx=0).to(device)
        self.Eh.weight.data = (1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double, device=device))
        
        self.rvh = torch.nn.Embedding(len(d.relations), dim, padding_idx=0).to(device)
        self.rvh.weight.data = (1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double, device=device))
        
        self.Wu = torch.nn.Parameter(torch.tensor(
            np.random.uniform(-1, 1, (len(d.relations), dim)),
            dtype=torch.double, requires_grad=True, device=device))
        
        self.bs = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device=device))
        self.bo = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device=device))
        
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, u_idx, r_idx, v_idx):
        # 确保索引张量在正确设备上
        u_idx = u_idx.to(self.device)
        r_idx = r_idx.to(self.device)
        v_idx = v_idx.to(self.device)
        
        u = self.Eh.weight[u_idx]
        v = self.Eh.weight[v_idx]
        Ru = self.Wu[r_idx]
        rvh = self.rvh.weight[r_idx]

        u = torch.where(torch.norm(u, 2, dim=-1, keepdim=True) >= 1, 
                        u/(torch.norm(u, 2, dim=-1, keepdim=True)-1e-5), u)
        v = torch.where(torch.norm(v, 2, dim=-1, keepdim=True) >= 1, 
                        v/(torch.norm(v, 2, dim=-1, keepdim=True)-1e-5), v)
        rvh = torch.where(torch.norm(rvh, 2, dim=-1, keepdim=True) >= 1, 
                          rvh/(torch.norm(rvh, 2, dim=-1, keepdim=True)-1e-5), rvh)   
        u_e = p_log_map(u)
        u_W = u_e * Ru
        u_m = p_exp_map(u_W)
        v_m = p_sum(v, rvh)
        u_m = torch.where(torch.norm(u_m, 2, dim=-1, keepdim=True) >= 1, 
                          u_m/(torch.norm(u_m, 2, dim=-1, keepdim=True)-1e-5), u_m)
        v_m = torch.where(torch.norm(v_m, 2, dim=-1, keepdim=True) >= 1, 
                          v_m/(torch.norm(v_m, 2, dim=-1, keepdim=True)-1e-5), v_m)
        
        sqdist = (2.*artanh(torch.clamp(torch.norm(p_sum(-u_m, v_m), 2, dim=-1), 1e-10, 1-1e-5)))**2

        return -sqdist + self.bs[u_idx] + self.bo[v_idx]

class MuRP_one_shot(torch.nn.Module):
    def __init__(self, d, dim):
        super(MuRP_one_shot, self).__init__()
        self.Eh = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.Eh.weight.data = (1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double, device=self.device))
        self.rvh = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rvh.weight.data = (1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double, device=self.device))
        self.Wu = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (len(d.relations), 
                                        dim)), dtype=torch.double, requires_grad=True, device=self.device))
        self.bs = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device=self.device))
        self.bo = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device=self.device))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, u_idx, r_idx, v_idx):
        u = self.Eh.weight[u_idx]
        v = self.Eh.weight[v_idx]
        Ru = self.Wu[r_idx]
        rvh = self.rvh.weight[r_idx]

        eps = 1e-5
        u_nrm = torch.linalg.vector_norm(u, 2, dim=-1, keepdim=True)
        v_nrm = torch.linalg.vector_norm(v, 2, dim=-1, keepdim=True)
        rvh_nrm = torch.linalg.vector_norm(rvh, 2, dim=-1, keepdim=True)
        u_nrm_msk = u_nrm >= 1
        v_nrm_msk = v_nrm >= 1
        rvh_nrm_msk = rvh_nrm >= 1
        u_nrm_e = u_nrm - eps
        v_nrm_e = v_nrm - eps
        rvh_nrm_e = rvh_nrm - eps

        u = (u/u_nrm_e) * u_nrm_msk + u * torch.logical_not(u_nrm_msk)
        v = (v/v_nrm_e) * v_nrm_msk + v * torch.logical_not(v_nrm_msk)
        rvh = (rvh/rvh_nrm_e) * rvh_nrm_msk + rvh * torch.logical_not(rvh_nrm_msk)

        u_e = p_log_map(u)
        u_W = u_e * Ru
        u_m = p_exp_map(u_W)
        v_m = p_sum(v, rvh)

        um_nrm = torch.linalg.vector_norm(u_m, 2, dim=-1, keepdim=True)
        vm_nrm = torch.linalg.vector_norm(v_m, 2, dim=-1, keepdim=True)
        um_nrm_msk = um_nrm >= 1
        vm_nrm_msk = vm_nrm >= 1
        um_nrm_e = um_nrm - eps
        vm_nrm_e = vm_nrm - eps
        u_m = (u_m / um_nrm_e) * um_nrm_msk + u_m * torch.logical_not(um_nrm_msk)
        v_m = (v_m / vm_nrm_e) * vm_nrm_msk + v_m * torch.logical_not(vm_nrm_msk)

        sqdist = (2.*artanh(torch.clamp(torch.linalg.vector_norm(p_sum(-u_m, v_m), 2, dim=-1), 1e-10, 1-1e-5)))**2

        return -sqdist + self.bs[u_idx] + self.bo[v_idx]

    def one_shot_encoding(self, v_idx, r_idx):
        v = self.Eh.weight[v_idx]
        Ru = self.Wu[r_idx]
        rvh = self.rvh.weight[r_idx]

        eps = 1e-5
        v_nrm = torch.linalg.vector_norm(v, 2, dim=-1, keepdim=True)
        rvh_nrm = torch.linalg.vector_norm(rvh, 2, dim=-1, keepdim=True)
        v_nrm_msk = v_nrm >= 1
        rvh_nrm_msk = rvh_nrm >= 1
        v_nrm_e = v_nrm - eps
        rvh_nrm_e = rvh_nrm - eps

        v = (v/v_nrm_e) * v_nrm_msk + v * torch.logical_not(v_nrm_msk)
        rvh = (rvh/rvh_nrm_e) * rvh_nrm_msk + rvh * torch.logical_not(rvh_nrm_msk)

        # We want u_m to be as close as possible to v after applying a relation-specific transformation.
        v_m = p_sum(torch.mean(v,1), torch.mean(rvh,1))

        return v_m

    def one_shot_encoding_avg(self, v_idx):
        v = self.Eh.weight[v_idx]
        # return the average v vector from all the terms in the definition
        return torch.mean(v,1)



class MuRE(torch.nn.Module):
    def __init__(self, d, dim):
        super(MuRE, self).__init__()
        self.E = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.E.weight.data = self.E.weight.data.double()
        self.device = torch.device("cuda" if self.cuda and torch.cuda.is_available() else "cpu")
        self.E.weight.data = (1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double, device=self.device))
        self.Wu = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (len(d.relations), 
                                        dim)), dtype=torch.double, requires_grad=True, device=self.device))
        self.rv = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rv.weight.data = self.rv.weight.data.double()
        self.rv.weight.data = (1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double, device=self.device))
        self.bs = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device=self.device))
        self.bo = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device=self.device))
        self.loss = torch.nn.BCEWithLogitsLoss()
       
    def forward(self, u_idx, r_idx, v_idx):
        u = self.E.weight[u_idx]
        v = self.E.weight[v_idx]
        Ru = self.Wu[r_idx]
        rv = self.rv.weight[r_idx]
        u_size = u.size()
        
        u_W = u * Ru

        sqdist = torch.sum(torch.pow(u_W - (v+rv), 2), dim=-1)
        return -sqdist + self.bs[u_idx] + self.bo[v_idx] 

