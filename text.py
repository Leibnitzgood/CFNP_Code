import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


# with open("./data/household_power_consumption/household_power_consumption.txt", "r") as f:
#     data = f.readlines()

# data = [d.strip().split(";") for d in data][1:]
# # 假设你的数据存储在 data 变量中
# # data = np.random.normal(size=100)  # 这只是一个示例，你应该使用你自己的数据

# # 直方图
# plt.hist(data, bins='auto')
# plt.show()

# # QQ图
# stats.probplot(data, plot=plt)
# plt.show()

# # Shapiro-Wilk 测试
# w, p = stats.shapiro(data)
# print(f"W = {w}, p = {p}")
import matplotlib.pyplot as plt
import numpy as np
import pickle

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

def plot_train_seq(train_seq, years, selected_year=None):
    """
    Plot the train_seq for the selected year. If no year is selected, plot all years.
    
    Parameters:
    - train_seq: List of training sequences.
    - years: List of years corresponding to each sequence.
    - selected_year: Year to plot. If None, plot all years.
    """
    if selected_year:
        indices = [i for i, year in enumerate(years) if year == selected_year]
        selected_seqs = [train_seq[i] for i in indices]
    else:
        selected_seqs = train_seq
    
    plt.figure(figsize=(12, 6))
    for seq in selected_seqs:
        plt.plot(seq)
    
    plt.title(f'Training Sequences{" for Year " + str(selected_year) if selected_year else ""}')
    plt.xlabel('Time')
    plt.ylabel('Sequence Value')
    plt.show()

# 加载数据并处理
with open("./data/symptom_data/saves/combine.pkl", "rb") as f:
    data = pickle.load(f)

include_cols = []  # 确定需要包含的列

regions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
start_year = 2018
year = 2022

train_val_years = [y for y in range(start_year, year)]
train_val_seqs = [get_sequence(data, hhs, y) for hhs in regions for y in train_val_years]

week_ahead = 1  # 预测的周数
train_dataset = [seq_to_dataset(seq, week_ahead) for seq in train_val_seqs]
X, X_symp, Y, wk, reg = [], [], [], [], []
for x, y, w, r in train_dataset:
    X.extend([l[:, -1] for l in x])
    X_symp.extend([l[:, :-1] for l in x])
    Y.extend(y)
    wk.extend(w)
    reg.extend(r)

# 使用获取的年份和数据序列绘图
plot_train_seq(X, train_val_years)
