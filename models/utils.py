import torch
import math
import torch.nn.functional as F
import numpy as np
from torch.distributions import Bernoulli
from itertools import product
from symp_extract.consts import include_cols

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
float_tensor = (
    torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
)
long_tensor = (
    torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
)

# 定义函数来获取数据序列
def get_sequence(data, hhs, year):
    d1 = data[(data.year == year - 1) & (data.hhs == hhs) & (data.epiweek > 20)][include_cols + ["ili", "epiweek", "hhs"]]
    d2 = data[(data.year == year) & (data.hhs == hhs) & (data.epiweek <= 20)][include_cols + ["ili", "epiweek", "hhs"]]
    d1 = np.array(d1)
    d2 = np.array(d2)
    return np.vstack((d1, d2))

def seq_to_dataset(seq, week_ahead):
    X, Y, wk, reg = [], [], [], []
    start_idx = max(week_ahead, seq.shape[0] - 32 + week_ahead)
    for i in range(start_idx, seq.shape[0]):
        X.append(seq[: i - week_ahead + 1, :-2])
        Y.append(seq[i, -3])
        wk.append(seq[i, -2])
        reg.append(seq[i, -1])
    return X, Y, wk, reg

def preprocess_seq_batch(seq_list: list):
    max_len = max([len(x) for x in seq_list])
    if len(seq_list[0].shape) == 2:
        ans = np.zeros((len(seq_list), max_len, len(seq_list[0][0])))
    else:
        ans = np.zeros((len(seq_list), max_len, 1))
        seq_list = [x[:, np.newaxis] for x in seq_list]
    for i, seq in enumerate(seq_list):
        ans[i, : len(seq), :] = seq
    return ans

def load_best_models(models, week, model_path='best_model/best_models_week{}.pth'):
    formatted_model_path = model_path.format(week)
    # Load the saved best model state dictionaries
    state_dicts = torch.load(formatted_model_path)
    
    # Iterate through the model list, loading each model's state
    for i, model in enumerate(models):
        model_name = f'model_{i}.pth'
        model.load_state_dict(state_dicts[model_name])
    print(f"Models loaded from: {formatted_model_path}")

#计算JSD
def kl_divergence(mu1, logstd1, mu2, logstd2):
    # Convert logstd to std
    std1 = torch.exp(logstd1)
    std2 = torch.exp(logstd2)
    
    kl = torch.log(std2 / std1) + (std1**2 + (mu1 - mu2)**2) / (2 * std2**2) - 0.5
    return kl.sum()

def jsd(mu1, logstd1, mu2, logstd2):
    m_mu = 0.5 * (mu1 + mu2)
    m_logstd = 0.5 * (logstd1 + logstd2)
    
    kl1 = kl_divergence(mu1, logstd1, m_mu, m_logstd)
    kl2 = kl_divergence(mu2, logstd2, m_mu, m_logstd)
    
    return 0.5 * kl1 + 0.5 * kl2

def logitexp(logp):
    # https://github.com/pytorch/pytorch/issues/4007
    pos = torch.clamp(logp, min=-0.69314718056)
    neg = torch.clamp(logp, max=-0.69314718056)
    neg_val = neg - torch.log(1 - torch.exp(neg))
    pos_val = -torch.log(torch.clamp(torch.expm1(-pos), min=1e-20))
    return pos_val + neg_val


def one_hot(x: torch.Tensor, n_classes=10):
    x_onehot = float_tensor(x.size(0), n_classes).zero_()
    x_onehot.scatter_(1, x[:, None], 1)

    return x_onehot


class LogitRelaxedBernoulli(object):
    def __init__(self, logits, temperature=0.3, **kwargs):
        self.logits = logits
        self.temperature = temperature

    def rsample(self):
        eps = torch.clamp(
            torch.rand(
                self.logits.size(), dtype=self.logits.dtype, device=self.logits.device
            ),
            min=1e-6,
            max=1 - 1e-6,
        )
        y = (self.logits + torch.log(eps) - torch.log(1.0 - eps)) / self.temperature
        return y

    def log_prob(self, value):
        return (
            math.log(self.temperature)
            - self.temperature * value
            + self.logits
            - 2 * F.softplus(-self.temperature * value + self.logits)
        )


class Normal(object):
    def __init__(self, means, logscales, **kwargs):
        self.means = means
        self.logscales = logscales

    def log_prob(self, value):
        log_prob = torch.pow(value - self.means, 2)
        log_prob *= -(1 / (2.0 * self.logscales.mul(2.0).exp()))
        log_prob -= self.logscales + 0.5 * math.log(2.0 * math.pi)
        return log_prob

    def sample(self, **kwargs):
        eps = torch.normal(
            float_tensor(self.means.size()).zero_(),
            float_tensor(self.means.size()).fill_(1),
        )
        return self.means + self.logscales.exp() * eps

    def rsample(self, **kwargs):
        return self.sample(**kwargs)


def order_z(z):
    # scalar ordering function
    if z.size(1) == 1:
        return z
    log_cdf = torch.sum(
        torch.log(0.5 + 0.5 * torch.erf(z / math.sqrt(2))), dim=1, keepdim=True
    )
    return log_cdf


def sample_DAG(Z, g, training=True, temperature=0.3):
    # get the indices of an upper triangular adjacency matrix that represents the DAG
    idx_utr = np.triu_indices(Z.size(0), 1)

    # get the ordering
    ordering = order_z(Z)
    # sort the latents according to the ordering
    sort_idx = torch.sort(torch.squeeze(ordering), 0)[1]
    Y = Z[sort_idx, :]
    # form the latent pairs for the edges
    Z_pairs = torch.cat([Y[idx_utr[0]], Y[idx_utr[1]]], 1)
    # get the logits for the edges in the DAG
    logits = g(Z_pairs)

    if training:
        p_edges = LogitRelaxedBernoulli(logits=logits, temperature=temperature)
        G = torch.sigmoid(p_edges.rsample())
    else:
        p_edges = Bernoulli(logits=logits)
        G = p_edges.sample()

    # embed the upper triangular to the adjacency matrix
    unsorted_G = float_tensor(Z.size(0), Z.size(0)).zero_()
    unsorted_G[idx_utr[0], idx_utr[1]] = G.squeeze()
    # unsort the dag to conform to the data order
    original_idx = torch.sort(sort_idx)[1]
    unsorted_G = unsorted_G[original_idx, :][:, original_idx]

    return unsorted_G


def sample_Clique(Z, g, training=True, temperature=0.3):
    # get the indices of an upper triangular adjacency matrix that represents the DAG
    # idx_utr = np.triu_indices(Z.size(0), 1)
    idx_utr = np.triu_indices(Z.size(0), 1)
    idx_ltr = np.triu_indices(Z.size(0), 1)
    idx_ltr = idx_ltr[1], idx_ltr[0]
    idx_utr = (
        np.concatenate([idx_utr[0], idx_ltr[0]]),
        np.concatenate([idx_utr[1], idx_ltr[1]]),
    )

    # get the ordering
    ordering = order_z(Z)
    # sort the latents according to the ordering
    sort_idx = torch.sort(torch.squeeze(ordering), 0)[1]
    Y = Z[sort_idx, :]
    # form the latent pairs for the edges
    Z_pairs = torch.cat([Y[idx_utr[0]], Y[idx_utr[1]]], 1)
    # get the logits for the edges in the DAG
    logits = g(Z_pairs)

    if training:
        p_edges = LogitRelaxedBernoulli(logits=logits, temperature=temperature)
        G = torch.sigmoid(p_edges.rsample())
    else:
        p_edges = Bernoulli(logits=logits)
        G = p_edges.sample()

    # embed the upper triangular to the adjacency matrix
    unsorted_G = float_tensor(Z.size(0), Z.size(0)).zero_()
    unsorted_G[idx_utr[0], idx_utr[1]] = G.squeeze()
    # unsort the dag to conform to the data order
    original_idx = torch.sort(sort_idx)[1]
    unsorted_G = unsorted_G[original_idx, :][:, original_idx]

    return unsorted_G


def bipartite(Z1, Z2, g):
    indices = []
    for element in product(range(Z1.size(0)), range(Z2.size(0))):
        indices.append(element)
    indices = np.array(indices)
    Z_pairs = torch.cat([Z1[indices[:, 0]], Z2[indices[:, 1]]], 1)

    # Get the logits representing the connection strengths
    logits = g(Z_pairs)

    # Instead of sampling, use the sigmoid function to get weights directly
    A_vals = torch.sigmoid(logits)

    # embed the values to the adjacency matrix
    A = torch.zeros(Z1.size(0), Z2.size(0), dtype=Z1.dtype, device=Z1.device)
    A[indices[:, 0], indices[:, 1]] = A_vals.squeeze()

    return A

class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        assert len(x.shape) > 1

        return x.view(x.shape[0], -1)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True,为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # Only save the model when validation loss improves
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, models):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        # 将所有模型的状态字典保存到一个文件中
        if val_loss < self.val_loss_min:
            state_dicts = {f'model_{i}.pth': model.state_dict() for i, model in enumerate(models)}
            torch.save(state_dicts, 'best_models.pth')
            self.val_loss_min = val_loss
       
    def get_val_loss_min(self):
        return self.val_loss_min

    
def compute_crps(y_true, mean_pred, std_pred):
    # 将NumPy数组转换为PyTorch张量，并确保其在适当的设备上
    def to_torch(x):
        if isinstance(x, torch.Tensor):
            return x.clone().detach()
        else:
            return torch.tensor(x).clone().detach()

    y_true, mean_pred, std_pred = map(to_torch, (y_true, mean_pred, std_pred))
    
    # 避免除以零
    std_pred = torch.clamp(std_pred, min=1e-8)
    
    # 计算高斯CRPS
    normed_diff = (y_true - mean_pred) / std_pred
    pdf = torch.exp(-0.5 * normed_diff ** 2) / (std_pred * math.sqrt(2 * math.pi))
    cdf = 0.5 * (1 + torch.erf(normed_diff / math.sqrt(2)))
    
    term1 = normed_diff * (2 * cdf - 1)
    term2 = 2 * pdf - 1 / math.sqrt(math.pi)
    
    crps = torch.mean(std_pred * (term1 + term2))
    
    return crps