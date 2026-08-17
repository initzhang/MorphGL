from MorphGL.utils import get_logger
mlog = get_logger()

import gc
import dgl
import math
import time
import torch
import psutil
import numpy as np
from numba import jit, prange

from third_party.salient.driver.dataset import FastDataset

process_id = psutil.Process()
mem = lambda pos: mlog(f"{pos} mem usage: {process_id.memory_info().rss/1024/1024/1024} GB")

@jit(nopython=True, parallel=True)
def csc_reorder(indptr, indices, new_degs, new_indptr, map_new2old, map_old2new):
    num_nodes = new_degs.shape[0]
    num_edges = indices.shape[0]
    new_indices = np.zeros_like(indices, dtype=np.int64)
    for i in prange(num_nodes):
        tmp_edges = new_degs[i]
        offset_new = new_indptr[i]
        offset_old = indptr[map_new2old[i]]
        for j in prange(tmp_edges):
            new_indices[offset_new + j] = map_old2new[indices[offset_old + j]]
    return new_indices

def construct_graph_from_arrays(indptr, indices, required_format='csc'):
    assert required_format in ['csc', 'csr']
    graph = dgl.graph((required_format, (indptr, indices, torch.Tensor())), num_nodes=indptr.shape[0]-1)
    graph = graph.formats(required_format)
    graph.pin_memory_()
    return graph

def inplace_pin_arrays(arrays):
    cudart = torch.cuda.cudart()
    for arr in arrays:
        assert isinstance(arr, torch.Tensor)
        torch.cuda.check_error(cudart.cudaHostRegister(arr.data_ptr(), arr.numel() * arr.element_size(), 0))
    mlog("finish pin arrays")
    # torch.cuda.check_error(cudart.cudaHostUnregister(arr.data_ptr())) 

def generate_stats(graph, train_idx):
    mlog("start calculate counts")
    fanouts, pre_epochs, bs = [5, 10, 15], 3, 1024
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    nfeat_counts = torch.zeros(graph.num_nodes()).cuda()
    adj_counts = torch.zeros(graph.num_nodes()).cuda()
    tic = time.time()
    for _ in range(pre_epochs):
        it = my_iter(bs, train_idx)
        for seeds in it:
            input_nodes, output_nodes, blocks = sampler.sample(graph, seeds)
            # for nfeat, each iteration we only need to prepare the input layer
            nfeat_counts[input_nodes] += 1
            # for adj, each iteration we need to access each block's dst nodes
            for block in blocks:
                dst_num = block.dstnodes().shape[0]
                cur_touched_adj = block.ndata[dgl.NID]['_N'][:dst_num]
                adj_counts[cur_touched_adj] += 1
    mlog(f"pre-sampling {pre_epochs} epochs time: {time.time()-tic:.3f}s")
    adj_counts = adj_counts.cpu()
    nfeat_counts = nfeat_counts.cpu()
    return adj_counts, nfeat_counts

def my_iter(bs, train_idx):
    pm = torch.randperm(train_idx.shape[0]).to(train_idx.device)
    local_train_idx = train_idx[pm]
    length = train_idx.shape[0] // bs
    for i in range(length):
        st = i * bs
        ed = (i+1) * bs
        yield local_train_idx[st:ed]

def fast_reorder_to_csc(old_row, old_col, nodes_perm):
    tic = time.time()
    num_nodes = old_row.shape[0]-1
    old_graph = dgl.graph(('csc', (old_row, old_col, torch.Tensor([]))), num_nodes=num_nodes)
    src, dst = old_graph.adj_sparse(fmt='coo')
    mmap = torch.zeros(nodes_perm.shape[0], dtype=torch.int64)
    mmap[nodes_perm] = torch.arange(nodes_perm.shape[0])
    src = mmap[src]
    dst = mmap[dst]
    new_graph = dgl.graph((src, dst), num_nodes=num_nodes)
    row, col, _ = new_graph.adj_sparse(fmt='csc')
    del src, dst, mmap, old_graph
    mlog(f"fast reorder time: {time.time()-tic:.3f}s")
    return row, col

def load_shared_data_without_reorder(dataset_name, dataset_root):
    """
    graph topo & feature data can be shared between DUCATI, DGL, and SALIENT
    DUCATI need to reorder graph
    """
    assert dataset_name in ['ogbn-products', 'ogbn-papers100M', 'twitter', 'uk']
    if dataset_name in ['twitter', 'uk']:
        dataset = FastDataset.from_dgl(name=dataset_name, root=dataset_root)
    else:
        dataset = FastDataset.from_path(dataset_root, dataset_name)
    train_idx = dataset.split_idx['train']
    num_classes = dataset.num_classes
    x = dataset.x
    y = dataset.y.unsqueeze(-1)
    row = dataset.rowptr
    col = dataset.col

    inplace_pin_arrays([x, y, row, col])
    mlog('finish loading shared data')
    return x, y, row, col, None, train_idx, num_classes


def load_shared_data(dataset_name, dataset_root):
    """
    graph topo & feature data can be shared between DUCATI, DGL, and SALIENT
    DUCATI need to reorder graph
    """
    #mem("0")
    # first load the original dataset
    assert dataset_name in ['ogbn-products', 'ogbn-papers100M', 'twitter', 'uk']
    if dataset_name in ['twitter', 'uk']:
        dataset = FastDataset.from_dgl(name=dataset_name, root=dataset_root)
    else:
        dataset = FastDataset.from_path(dataset_root, dataset_name)
    train_idx = dataset.split_idx['train']
    num_classes = dataset.num_classes
    x = dataset.x
    y = dataset.y.unsqueeze(-1)
    row = dataset.rowptr
    col = dataset.col

    counts = None
    #mem("1 (topo+nfeat)")
    # get counts
    num_nodes = x.shape[0]
    old_graph = dgl.graph(('csc', (row, col, torch.Tensor())), num_nodes=num_nodes)
    old_graph = old_graph.formats('csc')
    old_graph.pin_memory_()
    tmp_train_idx = train_idx.cuda()
    adj_counts, nfeat_counts = generate_stats(old_graph, tmp_train_idx)
    degs = old_graph.in_degrees()
    old_graph.unpin_memory_()
    del tmp_train_idx, old_graph
    gc.collect()

    #mem("2 (topo+nfeat+misc)")
    # reorder graph
    tic = time.time()
    priority = adj_counts/(degs+1)
    adj_order = priority.argsort(descending=True)

    mmap = np.zeros(num_nodes, dtype=np.int64)
    mmap[adj_order.numpy()] = np.arange(num_nodes)

    new_degs = degs[adj_order].numpy()
    new_indptr = np.zeros_like(row.numpy())
    new_indptr[1:] = new_degs.cumsum()

    new_indices = csc_reorder(row.numpy(), col.numpy(), new_degs, new_indptr, adj_order.numpy(), mmap)
    #mem("3 (2*topo+nfeat+misc)")
    del row, col, degs, new_degs, mmap

    row = torch.from_numpy(new_indptr)
    col = torch.from_numpy(new_indices)
    mlog(f"reorder time: {time.time()-tic:.3f}s")

    # reorder other ndata accordingly
    old_train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    old_train_mask[train_idx] = True
    new_train_mask = old_train_mask[adj_order]
    new_train_idx = new_train_mask.nonzero().reshape(-1)
    new_adj_counts = adj_counts[adj_order]
    new_nfeat_counts = nfeat_counts[adj_order]
    # Attention: we omit the reorder of x here since it is random
    #x = x[adj_order]
    y = y[adj_order]
    train_idx = new_train_idx
    torch.cuda.empty_cache()
    counts = (new_adj_counts, new_nfeat_counts)


    inplace_pin_arrays([x, y, row, col])
    mlog('finish loading shared data')
    return x, y, row, col, counts, train_idx, num_classes

def partition_train_idx(all_train_idx, ratio=0.5):
    """
    return: CPU_train_idx, GPU_train_idx
    """
    temp_train_idx = all_train_idx[torch.randperm(all_train_idx.shape[0])]
    sep = int(all_train_idx.shape[0] * ratio)
    mlog(f"split into two part, salient {sep} : dgl {all_train_idx.shape[0]-sep}")
    return temp_train_idx[:sep], temp_train_idx[sep:].cuda()
