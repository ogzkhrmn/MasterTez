import numpy as np
import torch
from ignite.engine import Engine
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable

import dataset


def compute_mean_std(engine, batch):
    b, c, *_ = batch['image'].shape
    data = batch['image'].reshape(b, c, -1).to(dtype=torch.float64)
    mean = torch.mean(data, dim=-1).sum(dim=0)
    mean2 = torch.mean(data ** 2, dim=-1).sum(dim=0)
    return {"mean": mean, "mean^2": mean2}


data = dataset.fit_data("alltrue.csv")
length = data.__len__()
channel_size = 12
n_inputs = len(data.columns) - 1

lstm1 = torch.load("mlmodels/10-10mae.model")
lstm1.eval()


data_groups = data.groupby('clientid')
compute_engine = Engine(compute_mean_std)
effect = [0] * 5
for key in data_groups.groups.keys():
    data = data_groups.get_group(key)
    data_last = data.drop(columns={"clientid"})
    x_train, y_train = dataset.get_data(data_last, n_inputs, channel_size)
    X_train_tensors_final = torch.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    outputs = lstm1(X_train_tensors_final)
    predict = np.array(outputs.cpu().data.numpy())
    true = np.array(y_train)
    for index, item in enumerate(predict):
        y_true = true[index]
        y_pred = predict[index]
        auc_score = roc_auc_score(y_true, y_pred)
        y_plot = y_pred[[0, 2, 6, 7, 8]]
        effect[0] = effect[0] + y_plot[0]
        effect[1] = effect[1] + y_plot[1]
        effect[2] = effect[2] + y_plot[2]
        effect[3] = effect[3] + y_plot[3]
        effect[4] = effect[4] + y_plot[4]
        # if auc_score < 0.80:
        #     print(y_true)
        #     print(y_pred)

amin, amax = min(effect), max(effect)
for i, val in enumerate(effect):
    effect[i] = (val - amin) / (amax - amin)

x = ['kanal1', 'kanal2', 'kanal3', 'kanal4', 'kanal5']
x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, np.asarray(effect))
plt.xticks(x_pos, x)
plt.xlabel("Reklam Kanalları", fontsize=15)
plt.ylabel("Kanal Etkisi",  fontsize=15)
plt.show()
