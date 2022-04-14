import numpy as np
import torch
from ignite.engine import Engine
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

import dataset
from CADM import CADM


def compute_mean_std(engine, batch):
    b, c, *_ = batch['image'].shape
    data = batch['image'].reshape(b, c, -1).to(dtype=torch.float64)
    mean = torch.mean(data, dim=-1).sum(dim=0)
    mean2 = torch.mean(data ** 2, dim=-1).sum(dim=0)
    return {"mean": mean, "mean^2": mean2}


data = dataset.fit_data("all.csv")
length = data.__len__()
channel_size = 13
num_epochs = 10

learning_rate = 0.001  # 0.001 lr
hidden_size = 10  # number of features in hidden state
num_layers = 1  # number of stacked lstm layers

n_inputs = len(data.columns) - 1
input_size = n_inputs - channel_size
train, test = train_test_split(data, test_size=0.33)
# x_train, y_train = dataset.get_data(train, n_inputs, channel_size)
# x_test, y_test = dataset.get_data(test, n_inputs, channel_size)

lstm1 = CADM(channel_size, input_size, hidden_size, num_layers)  # our lstm class
lstm1.cuda()
criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)

data_groups = train.groupby('clientid')
print("User Group Count", data_groups.ngroups)
lstm1.train()
for epoch in range(num_epochs):
    print(epoch)
    for key in data_groups.groups.keys():
        data = data_groups.get_group(key)
        data_last = data.drop(columns={"clientid"})
        x_train, y_train = dataset.get_data(data_last, n_inputs, channel_size)
        y_copy = y_train.copy()
        if y_train.iloc[0][channel_size - 1] == 0:
            y_copy.loc[:, :] = 0
        X_train_tensors = Variable(torch.Tensor(x_train).cuda())
        y_train_tensors = Variable(torch.Tensor(y_train.to_numpy()).cuda())
        X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
        outputs = lstm1.forward(X_train_tensors_final)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train_tensors)
        loss.backward()
        optimizer.step()

torch.save(lstm1, "alllstm.model")

lstm1.eval()
data_groups = test.groupby('clientid')
compute_engine = Engine(compute_mean_std)
auc_last = 0
times = 0
effect = [0] * 5
for key in data_groups.groups.keys():
    data = data_groups.get_group(key)
    data_last = data.drop(columns={"clientid"})
    x_train, y_train = dataset.get_data(data_last, n_inputs, channel_size)
    X_train_tensors = Variable(torch.Tensor(x_train).cuda())
    X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
    outputs = lstm1(X_train_tensors_final)
    predict = np.array(outputs.cpu().data.numpy())
    true = np.array(y_train)
    for index, item in enumerate(predict):
        y_true = true[index]
        y_pred = predict[index]
        auc_score = roc_auc_score(y_true, y_pred)
        auc_last += auc_score
        times += 1
        y_plot = y_pred[[0, 2, 6, 7, 8]]
        effect[0] = effect[0] + y_plot[0]
        effect[1] = effect[1] + y_plot[1]
        effect[2] = effect[2] + y_plot[2]
        effect[3] = effect[3] + y_plot[3]
        effect[4] = effect[4] + y_plot[4]
        # if auc_score < 0.80:
        #     print(y_true)
        #     print(y_pred)

print("Auc Score: ", auc_last / times)
