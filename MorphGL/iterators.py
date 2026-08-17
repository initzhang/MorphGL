from MorphGL.utils import get_logger, get_seeds_list
mlog = get_logger()

from third_party.salient.fast_trainer.samplers import *
from third_party.salient.fast_trainer.transferers import *
from third_party.ducati import DUCATI

import os
import dgl
import math
import torch
import queue
import atexit
from types import SimpleNamespace
from dgl.utils.pin_memory import gather_pinned_tensor_rows

def prepare_salient(x, y, row, col, train_idx, train_batch_size, num_workers, train_fanouts):
    if train_idx.shape[0] == 0:
        return Blank_iter()
    cfg = FastSamplerConfig(
        x=x, y=y,
        rowptr=row, col=col,
        idx=train_idx,
        batch_size=train_batch_size, sizes=train_fanouts,
        skip_nonfull_batch=False, pin_memory=True
    )
    mlog("after salient config")
    train_max_num_batches = min(400, cfg.get_num_batches())
    cpu_loader = FastSampler(num_workers, train_max_num_batches, cfg)
    mlog('SALIENT CPU batcher prepared')
    return cpu_loader

def prepare_ducati(graph, all_data, fanouts, train_idx, train_batch_size, counts, budgets):
    """
    ducati only support KH
    """
    if train_idx.shape[0] == 0:
        return Blank_iter()
    train_idx = train_idx.cuda()
    total_budget, adj_budget, nfeat_budget = budgets
    tmp_args = SimpleNamespace(total_budget=total_budget, adj_budget=adj_budget, nfeat_budget=nfeat_budget, 
            fanouts=",".join(map(str, fanouts)), fake_dim=all_data[0].shape[1], 
            pre_batches=200, bs=train_batch_size)
    all_data_ut = [dgl.contrib.UnifiedTensor(all_data[0], torch.device("cuda:0")),
            dgl.contrib.UnifiedTensor(all_data[1].reshape(-1), torch.device("cuda:0"))]
    if adj_budget == -1:
        mlog("start dual-cache allocation for DUCATI")
        seeds_list = get_seeds_list(tmp_args.pre_batches, train_batch_size, train_idx)
        adj_slope, nfeat_slope = DUCATI.get_slope(tmp_args, graph, counts, seeds_list, all_data_ut)
        tmp_args.adj_slope = adj_slope
        tmp_args.nfeat_slope = nfeat_slope
        cached_indptr, cached_indices, gpu_flag, gpu_map, all_cache = DUCATI.allocate_dual_cache(tmp_args, graph, all_data_ut, counts)
    else:
        mlog("use user-provided dual-cache configuration for DUCATI")
        cached_indptr, cached_indices = DUCATI.form_adj_cache(tmp_args, graph, counts)
        gpu_flag, gpu_map, all_cache, _ = DUCATI.form_nfeat_cache(tmp_args, all_data_ut, counts)

    mlog(f"current allocation plan: {tmp_args.adj_budget:.3f}GB adj cache & {tmp_args.nfeat_budget:.3f}GB nfeat cache")

    sampler = DUCATI.NeighborSampler(cached_indptr, cached_indices, fanouts)
    gpu_loader = DUCATI_iter(graph, sampler, all_data_ut, train_batch_size, train_idx, all_cache, gpu_flag, gpu_map)
    mlog('DUCATI batcher prepared')
    return gpu_loader


class PreparedBatch(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor
    adjs: List[DGLBlock]

    def record_stream(self, stream):
        self.x.record_stream(stream)
        self.y.record_stream(stream)
        # FIXME: old dgl version does not support record stream for heterograph
        #for adj in self.adjs:
        #    adj.record_stream(stream)

    def to(self, device, non_blocking=False):
        return PreparedBatch(
            x=self.x.to(
                device=device,
                non_blocking=non_blocking),
            y=self.y.to(
                device=device,
                non_blocking=non_blocking),
            adjs = [adj.to(device=device, non_blocking=non_blocking) 
                for adj in self.adjs]
        )

class Blank_iter(Iterator):
    def __init__(self):
        self.length = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        raise StopIteration

    def __len__(self):
        return self.length
       
class DUCATI_iter(Iterator):
    def __init__(self, graph, sampler, all_data, bs, train_idx, all_cache, gpu_flag, gpu_map):
        self.graph = graph
        self.sampler = sampler
        self.all_data = all_data
        self.bs = bs
        self._idx = train_idx.cuda()
        self.all_cache = all_cache
        self.gpu_flag = gpu_flag
        self.gpu_map = gpu_map
        self.length = math.ceil(self._idx.shape[0] / self.bs)
        self.nfeat_buf = None
        self.label_buf = None
        self.with_cache = True

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, new_idx: torch.Tensor):
        self._idx = new_idx.cuda()
        self.length = math.ceil(self._idx.shape[0] / self.bs)

    def load_from_cache(self, cpu_partial, gpu_orig, idx, out_buf):
        gpu_mask = self.gpu_flag[idx]
        gpu_nids = idx[gpu_mask]
        gpu_local_nids = self.gpu_map[gpu_nids].long()
        cur_res = out_buf[:idx.shape[0]]
        cur_res[gpu_mask] = gpu_orig[gpu_local_nids].reshape(-1,cur_res.shape[1])
        cur_res[~gpu_mask] = cpu_partial.reshape(-1,cur_res.shape[1])
        #if gpu_orig.dim() == 2:
        #    mlog(f"nfeat hit: {gpu_local_nids.shape[0] * gpu_orig.shape[1] * gpu_orig.element_size() / 1024**2:.2f} MB")
        #else:
        #    mlog(f"label hit: {gpu_local_nids.shape[0] * gpu_orig.element_size() / 1024**2:.2f} MB")
        return cur_res

    def fetch_partial_batch(self, partial_batch):
        input_nodes, cur_bs, cpu_x, cpu_y, blocks = partial_batch
        if self.nfeat_buf is None:
            # create
            self.nfeat_buf = torch.zeros((int(1.5*input_nodes.shape[0]), cpu_x.shape[1]), dtype=cpu_x.dtype, device=torch.device('cuda:0'))
            self.label_buf = torch.zeros((self.bs, 1), dtype=cpu_y.dtype, device=torch.device('cuda:0'))

        if self.nfeat_buf.shape[0] < input_nodes.shape[0]:
            # resize
            mlog('resizing buffer')
            del self.nfeat_buf, self.label_buf
            self.nfeat_buf = torch.zeros((int(1.2*input_nodes.shape[0]), cpu_x.shape[1]), dtype=cpu_x.dtype, device=torch.device('cuda:0'))
            self.label_buf = torch.zeros((self.bs, 1), dtype=cpu_y.dtype, device=torch.device('cuda:0'))

        ret_x = self.load_from_cache(cpu_x, self.all_cache[0], input_nodes, self.nfeat_buf)
        ret_y = self.load_from_cache(cpu_y, self.all_cache[1], input_nodes[:cur_bs], self.label_buf)
        return ret_x, ret_y, blocks

    def __iter__(self):
        self.pos = 0
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        if self.pos == self.length:
            raise StopIteration
        st = self.pos * self.bs
        ed = min(self._idx.shape[0], st + self.bs)
        self.pos += 1
        cur_seeds = self._idx[st:ed]
        input_nodes, output_nodes, blocks = self.sampler.sample(self.graph, cur_seeds)
        # only fetch partial data on CPU, leave the GPU part for later process
        cpu_mask = ~self.gpu_flag[input_nodes]
        #cpu_x = gather_pinned_tensor_rows(self.all_data[0], input_nodes[cpu_mask])
        #cpu_y = gather_pinned_tensor_rows(self.all_data[1], output_nodes[cpu_mask[:(ed-st)]])
        cpu_x = self.all_data[0][input_nodes[cpu_mask]]
        cpu_y = self.all_data[1][output_nodes[cpu_mask[:(ed-st)]]]

        # partial batch format: input_nodes, batch_size, cpu_x, cpu_y, blocks
        return input_nodes, ed-st, cpu_x, cpu_y, blocks

