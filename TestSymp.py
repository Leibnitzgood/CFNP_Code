import pickle
from optparse import OptionParser
import networkx as nx

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from symp_extract.consts import include_cols
import dgl
from functools import reduce
import math
import random
from tqdm import tqdm
from symp_extract.utils import epiweek_to_month
from symp_extract.consts import hhs_neighbors
from properscoring import crps_gaussian

from models.utils import (
    device, 
    float_tensor, 
    long_tensor, 
    EarlyStopping, 
    seq_to_dataset, 
    get_sequence, 
    preprocess_seq_batch, 
    load_best_models,
)
from models.model import (
    EmbedEncoder,
    GRUEncoder,
    EmbGCNEncoder,
    LatentEncoder,
    CorrEncoder,
    Decoder,
    create_corr_encoder
)

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
 
seed_everything(42)

parser = OptionParser()
parser.add_option("-w", "--week", dest="week_ahead", type="int", default=1)
parser.add_option("-y", "--year", dest="year", type="int", default=2022)
parser.add_option("-n", "--num", dest="num", type="string")
parser.add_option("-e", "--epoch", dest="epochs", type="int", default="3000")
parser.add_option('--learning-rate', dest="Ir", type=float, default=5e-3, help='learning rate')
parser.add_option('--weight-decay', dest="Wd", type=float, default=1e-5, help='weight decay')
(options, args) = parser.parse_args()
 
regions = list(range(1, 11))
week_ahead = options.week_ahead 
num = options.num
epochs = options.epochs
year = options.year
Ir_number = options.Ir
weight_decay_number = options.Wd

with open("./data/symptom_data/saves/combine.pkl", "rb") as f:
    data = pickle.load(f) 

# 确定训练年份、验证年份和测试年份
start_year = 2018
year = 2022
train_years = [y for y in range(start_year, year)]
test_years = [year]

# 获取训练数据序列
train_val_years = train_years
train_val_seqs = [get_sequence(data, hhs, year) for hhs in regions for year in train_val_years]

# 按顺序划分训练集和验证集
val_frac = 0.2  # 验证集所占比例
num_train_val = int(len(train_val_seqs) * (1 - val_frac))
train_indices = list(range(0, num_train_val))
val_indices = list(range(num_train_val, len(train_val_seqs)))
train_seqs = [train_val_seqs[i] for i in train_indices]
val_seqs = [train_val_seqs[i] for i in val_indices]
test_seqs = [get_sequence(data, hhs, year) for hhs in regions for year in test_years]
train_dataset = [seq_to_dataset(seq, week_ahead) for seq in train_seqs]
val_dataset = [seq_to_dataset(seq, week_ahead) for seq in val_seqs]
test_dataset = [seq_to_dataset(seq, week_ahead) for seq in test_seqs]

X, X_symp, Y, wk, reg = [], [], [], [], []
for x, y, w, r in train_dataset:
    X.extend([l[:, -1] for l in x])
    X_symp.extend([l[:, :-1] for l in x])
    Y.extend(y)
    wk.extend(w)
    reg.extend(r)

X_val, X_symp_val, Y_val, wk_val, reg_val = [], [], [], [], []
for x, y, w, r in val_dataset:
    X_val.extend([l[:, -1] for l in x])
    X_symp_val.extend([l[:, :-1] for l in x])
    Y_val.extend(y)
    wk_val.extend(w)
    reg_val.extend(r)

X_test, X_symp_test, Y_test, wk_test, reg_test = [], [], [], [], []
for x, y, w, r in test_dataset:
    X_test.extend([l[:, -1] for l in x])
    X_symp_test.extend([l[:, :-1] for l in x])
    Y_test.extend(y)
    wk_test.extend(w)
    reg_test.extend(r)

# Convert Epiweek to month
mt = [epiweek_to_month(w) - 1 for w in wk]
mt_val = [epiweek_to_month(w) - 1 for w in wk_val]
mt_test = [epiweek_to_month(w) - 1 for w in wk_test]

# Get HHS adjacency graph
adj = nx.Graph()
adj.add_nodes_from(regions)


for i in range(1, len(regions)):
    adj.add_edges_from([(i, j) for j in hhs_neighbors[i]])

graph = dgl.from_networkx(adj)
graph = dgl.add_self_loop(graph)

# Reference points
seq_references = [x[:, -3] for x in train_seqs]
symp_references = [x[:, :-3] for x in train_seqs]
month_references = np.arange(12)
reg_references = np.array(regions) - 1.0


seq_references = preprocess_seq_batch(seq_references)
symp_references = preprocess_seq_batch(symp_references)

train_seqs = preprocess_seq_batch(X)
train_y = np.array(Y)
train_symp_seqs = preprocess_seq_batch(X_symp)
mt = np.array(mt, dtype=np.int32)
mt_test = np.array(mt_test, dtype=np.int32)
mt_val = np.array(mt_val, dtype=np.int32)
reg = np.array(reg, dtype=np.int32) - 1
reg_test = np.array(reg_test, dtype=np.int32) - 1
reg_val = np.array(reg_val, dtype=np.int32) - 1
test_seqs = preprocess_seq_batch(X_test)
test_symp_seqs = preprocess_seq_batch(X_symp_test)
test_y = np.array(Y_test)
val_seqs = preprocess_seq_batch(X_val)
val_symp_seqs = preprocess_seq_batch(X_symp_val)    
val_y = np.array(Y_val)

month_enc = EmbedEncoder(in_size=12, out_dim=60).to(device)
seq_encoder = GRUEncoder(in_size=1, out_dim=60).to(device)
symp_encoder = GRUEncoder(in_size=14, out_dim=60).to(device)
reg_encoder = EmbGCNEncoder(in_size=11, emb_dim=60, out_dim=60, num_layers=2, device=device).to(device)
stoch_month_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
stoch_seq_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
stoch_symp_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
stoch_reg_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
month_corr = create_corr_encoder(device)
seq_corr = create_corr_encoder(device)
symp_corr = create_corr_encoder(device)
reg_corr = create_corr_encoder(device)
decoder = Decoder(z_dim=60, sr_dim=60, latent_dim=60, hidden_dim=60, y_dim=1).to(device)

models = [
    month_enc,
    seq_encoder,
    symp_encoder,
    reg_encoder,
    stoch_month_enc,
    stoch_seq_enc,
    stoch_symp_enc,
    stoch_reg_enc,
    month_corr,
    seq_corr,
    symp_corr,
    reg_corr,
    decoder,
]

opt = optim.Adam(
    reduce(lambda x, y: x + y, [list(m.parameters()) for m in models]), lr = Ir_number, weight_decay = weight_decay_number
)


def evaluate(test_seqs, test_symp_seqs, reg_test, mt_test, test_y, sample=True, print1=True):
    for m in models:
        m.eval()
    # Porbabilistic encode of reference points
    ref_months = month_enc.forward(long_tensor(month_references))
    ref_seq = seq_encoder.forward(float_tensor(seq_references))
    ref_symp = symp_encoder.forward(float_tensor(symp_references))
    ref_reg = reg_encoder.forward(long_tensor(reg_references), graph.to(device))

    stoch_ref_months = stoch_month_enc.forward(ref_months)[0]
    stoch_ref_seq = stoch_seq_enc.forward(ref_seq)[0]
    stoch_ref_symp = stoch_symp_enc.forward(ref_symp)[0]
    stoch_ref_reg = stoch_reg_enc.forward(ref_reg)[0]

    # Probabilistic encode of test points

    test_months = month_enc.forward(long_tensor(mt_test.astype(int)))
    test_seq = seq_encoder.forward(float_tensor(test_seqs))
    test_symp = symp_encoder.forward(float_tensor(test_symp_seqs))
    test_reg = torch.stack([ref_reg[i] for i in reg_test], dim=0)

    stoch_test_months = stoch_month_enc.forward(test_months)[0]
    stoch_test_seq = stoch_seq_enc.forward(test_seq)[0]
    stoch_test_symp = stoch_symp_enc.forward(test_symp)[0]
    stoch_test_reg = stoch_reg_enc.forward(test_reg)[0]
    # Get view-aware latent embeddings
    test_months_z, test_month_sr, _, _, _, _ = month_corr.predict(
        stoch_ref_months, stoch_test_months, ref_months, test_months
    )
    test_seq_z, test_seq_sr, _, _, _, _ = seq_corr.predict(
        stoch_ref_seq, stoch_test_seq, ref_seq, test_seq
    )
    test_symp_z, test_symp_sr, _, _, _, _ = symp_corr.predict(
        stoch_ref_symp, stoch_test_symp, ref_symp, test_symp
    )
    test_reg_z, test_reg_sr, _, _, _, _ = reg_corr.predict(
        stoch_ref_reg, stoch_test_reg, ref_reg, test_reg
    )

    # Concat all latent embeddings
    test_z = torch.stack([test_months_z, test_seq_z, test_symp_z, test_reg_z], dim=1)
    test_sr = torch.stack(
        [test_month_sr, test_seq_sr, test_symp_sr, test_reg_sr], dim=1
    )

    sample_y, mean_y, _, _, std_y = decoder.predict(
        test_z, test_sr, test_seq, sample=sample
    )
    sample_y = sample_y.detach().cpu().numpy().ravel()
    mean_y = mean_y.detach().cpu().numpy().ravel()
    std_y = std_y.detach().cpu().numpy().ravel()

    # RMSE loss
    smooth_term = 1e-10
    squared_diff = (sample_y - test_y.ravel()) ** 2
    mean_squared_error = np.mean(squared_diff)
    rmse = np.sqrt(mean_squared_error + smooth_term)
    crps_value = crps_gaussian(test_y.ravel(), mean_y, std_y)
    if print1 == True:
        print(f"RMSE = {rmse}")
        print(f"CRPS = {crps_value}")
    else:
        pass
    return rmse, mean_y, sample_y, std_y, crps_value


# Load best models
load_best_models(models, week_ahead)

# Define function to evaluate multiple times
def evaluate_multiple_times(test_data, num_trials):
    rmse_values = []
    crps_values = []
    sample_y_value = []

    for _ in range(num_trials):
        test_rmse, _, sample_y, _, test_crps_value = evaluate(*test_data, sample=True, print1=False)
        rmse_values.append(test_rmse)
        crps_values.append(test_crps_value)
        sample_y_value = sample_y

    return rmse_values, crps_values, sample_y_value

# Prepare test data
test_data = (test_seqs, test_symp_seqs, reg_test, mt_test, test_y)
num_trials = 20
rmse_values, crps_values, sample_y = evaluate_multiple_times(test_data, num_trials)

# Compute mean and standard deviation
mean_rmse = np.mean(rmse_values)
mean_crps = np.mean(crps_values)

# Print statistics
print(f"Test RMSE: {mean_rmse:.2f}")
print(f"Test CRPS: {mean_crps:.2f}")

