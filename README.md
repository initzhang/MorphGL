# MorphGL

## Installation

The experiments are conducted with libraries compiled with CUDA 11.7 and gcc9.3.
```
# basic setup
conda create -n morph python=3.9
conda activate morph
pip install torch==2.0.1 numpy==1.26.4 pandas==2.2.2 packaging ogb==1.3.6 numba==0.59.1

# install customized dgl of ducati
git clone https://github.com/initzhang/dc_dgl.git
cd dc_dgl
git checkout mix
sh mybuild.sh

# install pyg required by salient
pip install torch_geometric torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# install salient
cd third_party/salient/fast_sampler
python setup.py install
```

## Running Experiment

Run a specific experiment with: `python example_usage.py --num_workers 11 --total_budget 20 --model sage --hidden_features 256`.
You can see all the arguments in `parser.py`.


MorphGL: total cache budget configurations used in Table 4 (in GB)

|      | Machine A | Machine B | Machine C |
| ---- | --------- | --------- | --------- |
|      |           | GCN       |           |
| PA   | 18        | 18        | 6         |
| TW   | 17        | 17        | 5         |
| UK   | 18        | 18        | 6         |
|      |           | GraphSAGE |           |
| PA   | 18        | 18        | 6         |
| TW   | 17        | 17        | 5         |
| UK   | 18        | 18        | 6         |
|      |           | GAT       |           |
| PA   | 18        | 18        | 6         |
| TW   | 17        | 17        | 5         |
| UK   | 18        | 18        | 6         |

DUCATI:  total cache budget configurations used in Table 4 (in GB)

|      | Machine A | Machine B | Machine C |
| ---- | --------- | --------- | --------- |
|      |           | GCN       |           |
| PA   | 20        | 20        | 8         |
| TW   | 20        | 20        | 8         |
| UK   | 20        | 20        | 8         |
|      |           | GraphSAGE |           |
| PA   | 20        | 20        | 8         |
| TW   | 20        | 20        | 8         |
| UK   | 20        | 20        | 8         |
|      |           | GAT       |           |
| PA   | 20        | 20        | 8         |
| TW   | 20        | 20        | 8         |
| UK   | 20        | 20        | 8         |

