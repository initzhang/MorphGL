import dgl
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics.functional as MF

from mylog import get_logger
mlog = get_logger()

import DUCATI
from model import SAGE
from load_graph import load_acc
from common import set_random_seeds, get_epoch_seeds

def entry(args, graph, all_data, train_idx, val_idx, counts):

    # prepare two cache
    fanouts = [int(x) for x in args.fanouts.split(",")]
    cached_indptr, cached_indices = DUCATI.CacheConstructor.form_adj_cache(args, graph, counts)
    sampler = DUCATI.NeighborSampler(cached_indptr, cached_indices, fanouts)
    gpu_flag, gpu_map, all_cache, _ = DUCATI.CacheConstructor.form_nfeat_cache(args, all_data, counts)

    # prepare a buffer
    rand_idxs = torch.randint(0, train_idx.shape[0], (args.bs,))
    input_nodes, _, _ = sampler.sample(graph, train_idx[rand_idxs])
    estimate_max_batch = int(1.2*input_nodes.shape[0])
    nfeat_buf = torch.zeros((estimate_max_batch, all_data[0].shape[1]), dtype=torch.float).cuda()
    label_buf = torch.zeros((args.bs, ), dtype=torch.long).cuda()
    mlog(f"buffer size: {(estimate_max_batch*all_data[0].shape[1]*4+args.bs*8)/(1024**3):.3f} GB")

    nfeat_loader = DUCATI.NfeatLoader(all_data[0], all_cache[0], gpu_map, gpu_flag)
    label_loader = DUCATI.NfeatLoader(all_data[1], all_cache[1], gpu_map, gpu_flag)

    # prepare model
    if args.model == 'sage':
        model = SAGE(all_data[0].shape[1], args.num_hidden, args.n_classes, len(fanouts), F.relu, args.dropout)
    else:
        from gcn_model import GCN
        model = GCN(all_data[0].shape[1], args.num_hidden, args.n_classes, len(fanouts), F.relu, args.dropout)
    model = model.cuda()
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # for validation
    if args.valfan == '0':
        valfan = [-1,-1,-1]
    else:
        valfan = [int(x) for x in args.valfan.split(",")]
    val_sampler = dgl.dataloading.NeighborSampler(valfan)

    epoch_times = []
    epoch_accs = []
    for e in range(1,args.epochs+1):
        #######
        # train
        #######
        tic = time.time()
        model.train()
        total_loss = 0
        cur_train_seeds = get_epoch_seeds(args.bs, train_idx)
        for seeds in cur_train_seeds:
            # Adj-Sampling
            input_nodes, output_nodes, blocks = sampler.sample(graph, seeds)
            # Nfeat-Selecting
            cur_nfeat = nfeat_loader.load(input_nodes, nfeat_buf) # fetch nfeat
            cur_label = label_loader.load(output_nodes, label_buf) # fetch label
            # train
            batch_pred = model(blocks, cur_nfeat)
            loss = loss_fcn(batch_pred, cur_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        time_cur_epoch = time.time() - tic
        epoch_times.append(time_cur_epoch)
        mlog(f'{e:03d}-th epoch Training time: {time_cur_epoch:.4f}s, loss: {total_loss:.4f}')

        #######
        # valid
        #######
        tic = time.time()
        model.eval()
        ys = []
        preds = []
        cur_val_seeds = get_epoch_seeds(args.valbs, val_idx)
        for seeds in cur_val_seeds:
            input_nodes, output_nodes, blocks = val_sampler.sample(graph, seeds)
            ys.append(all_data[1][output_nodes]) # fetch label
            cur_nfeat = all_data[0][input_nodes] # fetch nfeat
            with torch.no_grad():
                outputs = model(blocks, cur_nfeat)
                pred = torch.topk(outputs, k=1).indices.view(-1)
                preds.append(pred)
        if args.metric == 'acc':
            cur_acc = MF.accuracy(torch.cat(preds), torch.cat(ys))
            epoch_accs.append(cur_acc.item())
            #cur_acc = ((torch.cat(preds) == torch.cat(ys)) + 0.0).sum().item() / val_idx.shape[0]
            #epoch_accs.append(cur_acc)
        else:
            cur_acc = MF.f1_score(torch.cat(preds), torch.cat(ys))
            epoch_accs.append(cur_acc.item())
        time_cur_valid = time.time() - tic
        mlog(f'{e:03d}-th epoch Validation time: {time_cur_valid:.4f}s, val {args.metric}: {cur_acc:.4f}')

    #mlog(f"ducati: {args.adj_budget:.3f}GB adj cache & {args.nfeat_budget:.3f}GB nfeat cache time: {np.mean(avgs):.2f} ± {np.std(avgs):.2f}ms\n")
    mlog(epoch_times)
    mlog(epoch_accs)

    mlog(f"peak val {args.metric}: {max(epoch_accs)}")
    mlog(f"average epoch time: {np.mean(epoch_times):.2f} ± {np.std(epoch_times):.2f} s")

def new_separate(args, graph, train_mask, val_mask, label, feat):
    separate_tic = time.time()
    train_idx = torch.nonzero(train_mask).reshape(-1)
    val_idx = torch.nonzero(val_mask).reshape(-1)
    mlog(f"training nodes: {train_idx.shape[0]}, validation nodes: {val_idx.shape[0]}")

    # cleanup
    graph.ndata.clear()
    graph.edata.clear()

    # pin
    nfeat = dgl.contrib.UnifiedTensor(feat.float(), device='cuda')
    label = dgl.contrib.UnifiedTensor(label.long(), device='cuda')

    mlog(f'finish pinning features/labels/graph, time elapsed: {time.time()-separate_tic:.2f}s')
    return graph, [nfeat, label], train_idx, val_idx



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset params
    parser.add_argument("--dataset", type=str, choices=['ogbn-papers100M', 'ogbn-products'],
                        default='ogbn-papers100M')
                        #default='ogbn-products')
    parser.add_argument("--pre-epochs", type=int, default=2) # PreSC params

    # running params
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--nfeat-budget", type=float, default=0.2) # in GB
    parser.add_argument("--adj-budget", type=float, default=0.1) # in GB
    parser.add_argument("--bs", type=int, default=1000)
    parser.add_argument("--fanouts", type=str, default='15,10,5')
    parser.add_argument("--pre-batches", type=int, default=100)

    parser.add_argument("--valbs", type=int, default=100)
    parser.add_argument("--valfan", type=str, default='15,15,15')

    # gnn model params
    parser.add_argument("--model", type=str, choices=['sage', 'gcn'], default='sage')
    parser.add_argument('--num-hidden', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.003)

    parser.add_argument("--metric", type=str, choices=['f1', 'acc'], default='acc')

    args = parser.parse_args()
    mlog(args)
    #set_random_seeds(0)

    # dataload & pinning
    graph, n_classes, train_mask, val_mask, label, feat, counts = load_acc(args)
    args.n_classes = n_classes
    graph, all_data, train_idx, val_idx = new_separate(args, graph, train_mask, val_mask, label, feat)
    #print(all_data)
    train_idx = train_idx.cuda()
    val_idx = val_idx.cuda()
    if args.model == 'gcn':
        graph = dgl.add_self_loop(graph)
        graph.create_formats_()
    graph.pin_memory_()
    mlog(graph)

    entry(args, graph, all_data, train_idx, val_idx, counts)
