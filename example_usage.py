import os
import gc
import time
import torch
import psutil
import numpy as np
import torch.nn as nn
import torch.optim as optim

import MorphGL
from models.gsg import SAGE
from models.gat import GAT
from models.gcn import GCN
from parser import make_parser
from MorphGL.iterators import prepare_salient, prepare_ducati
from loaders import load_shared_data, construct_graph_from_arrays


if __name__ == "__main__":
    MorphGL.utils.set_seeds(0)
    mlog = MorphGL.utils.get_logger()
    args = make_parser().parse_args()
    cur_process = psutil.Process(os.getpid())
    mlog(f'CMDLINE: {" ".join(cur_process.cmdline())}')
    mlog(args)

    # load shared data
    x, y, row, col, counts, train_idx, num_classes = load_shared_data(args.dataset_name, args.dataset_root)
    graph = construct_graph_from_arrays(row, col)
    train_idx = train_idx[torch.randperm(train_idx.shape[0])]

    if args.baseline in ["salient"]:
        cpu_train_idx = train_idx
        gpu_train_idx = torch.tensor([]).cuda()
    elif args.baseline in ["ducati"]:
        cpu_train_idx = torch.tensor([])
        gpu_train_idx = train_idx.cuda()
    else:
        assert args.baseline == ''
        cpu_train_idx = train_idx
        gpu_train_idx = train_idx.cuda()

    # prepare GPU batcher
    gpu_loader = prepare_ducati(graph, (x, y), args.train_fanouts, 
            gpu_train_idx, args.train_batch_size, counts, 
            (args.total_budget, args.adj_budget, args.nfeat_budget))

    # prepare CPU batcher
    cpu_loader = prepare_salient(x, y, row, col, cpu_train_idx, 
            args.train_batch_size, args.num_workers, args.train_fanouts[::-1])

    # prepare model etc
    if args.model == 'sage':
        model = SAGE(x.shape[1], args.hidden_features, num_classes, len(args.train_fanouts)).cuda()
    elif args.model == 'gat':
        model = GAT(x.shape[1], args.hidden_features, num_classes, len(args.train_fanouts)).cuda()
    elif args.model == 'gcn':
        model = GCN(x.shape[1], args.hidden_features, num_classes, len(args.train_fanouts)).cuda()
    else:
        raise ValueError
    mlog(model)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # prepare input dict
    input_dict = {}
    input_dict["GPU_id"] = 0
    input_dict["CPU_cores"] = args.num_workers
    input_dict["CPU_loader"] = cpu_loader
    input_dict["GPU_loader"] = gpu_loader
    input_dict["dataset"] = (x, y, row, col, graph, train_idx, num_classes)
    input_dict["model"] = (model, loss_fcn, optimizer)

    if args.baseline.strip():
        #################################
        # run baseline 
        #################################
        if args.baseline.strip() in ['salient']:
            sched_plan = ('naive_pipe', 0, 0)
        else:
            assert args.baseline.strip() in ['ducati']
            sched_plan = ('no_pipe', 0, 0)
    else:
        #################################
        # run MorphGL
        #################################
        if args.profs == '':
            # rerun profiling
            MorphGL.Profiler(args.trials, input_dict)
        else:
            # use provided profiling infos
            t_cache, t_gpu, t_cpu, t_dma, t_model, total_batches = [float(x.strip()) for x in args.profs.split(",")]
            MorphGL.prof_infos = t_cache, t_gpu, t_cpu, t_dma, t_model, int(total_batches)

        # decide the partition and schedule plan
        partition_plan, feedback = None, None
        while True:
            partition_plan = MorphGL.Partitioner(partition_plan, feedback)
            feedback, sched_plan, converge = MorphGL.Scheduler(partition_plan, args.buffer_size)
            if converge:
                break

    mlog(sched_plan)
    gc.collect()
    torch.cuda.empty_cache()
    trainer = MorphGL.Executor(input_dict, sched_plan)
    # train
    durs = []
    for r in range(args.epochs):
        mlog(f'\n\n==============')
        mlog(f'RUN {r}')
        torch.cuda.synchronize()
        tic = time.time()
        trainer.train_one_epoch()
        torch.cuda.synchronize()
        dur = time.time() - tic
        mlog(dur)
        durs.append(dur)
        torch.cuda.empty_cache()
    mlog(durs)
    mlog(f"averaged epoch time: {np.mean(durs[1:]):.2f} ± {np.std(durs[1:]):.2f}")
