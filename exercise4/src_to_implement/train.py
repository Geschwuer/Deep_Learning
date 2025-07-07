import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data import ChallengeDataset
from model import Model
from trainer import Trainer

# load the data from the csv file and perform a train-test-split
df = pd.read_csv("data.csv", sep=";")
train, val = train_test_split(df, test_size=0.2, random_state=42)
# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_ds = ChallengeDataset(data=train, mode="train")
val_ds = ChallengeDataset(data=val, mode="val")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# create an instance of our ResNet model
model = Model(num_classes=2)
# set up loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# set up loss function
# use weighting to tackle class imbalance
class_counts = df[['crack', 'inactive']].sum().values
total = len(df)
freqs = class_counts / total
pos_weight = (1.0 - freqs) / freqs
criterium = nn.BCEWithLogitsLoss(pos_weight=t.Tensor(pos_weight))

# create trainer instance
trainer = Trainer(
    model=model,
    crit=criterium,
    optim=optimizer,
    train_dl=train_loader,
    val_test_dl=val_loader,
    cuda=t.cuda.is_available(),
    early_stopping_patience=7
)

# go, go, go... call fit on trainer
res = trainer.fit(epochs=50)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')