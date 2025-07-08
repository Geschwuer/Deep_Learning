def run_training(
       epochs = 40,
       lr = 1e-3,
       bs = 32,
       weight_decay = 1e-4,
       early_stopping_patience = 5,
       use_augmentation = True,
       model_save_dir = "trainings",
       seed = 42 
):

    import torch as t
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from data import DataAugmenter
    import seaborn as sns
    import datetime
    from pathlib import Path

    from data import ChallengeDataset
    from model import Model
    from trainer import Trainer

    # load the data from the csv file and perform a train-test-split
    df = pd.read_csv("data.csv", sep=";")
    train, val = train_test_split(df, test_size=0.2, random_state=seed)

    if use_augmentation:
        augmenter = DataAugmenter(train)
        train = augmenter.balance_dataset()
        train.to_csv("augmented_data.csv", index = False)

    # class distributio
    df["class_combo"] = df.apply(lambda row: f"{row['crack']}_{row['inactive']}", axis=1)
    sns.countplot(x="class_combo", data=df)
    plt.title("class distribution after augmentation")
    plt.xlabel("crack | inactive")
    plt.ylabel("number of samples")
    plt.savefig('DataAugmentation.png')

    # set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
    train_ds = ChallengeDataset(data=train, mode="train")
    val_ds = ChallengeDataset(data=val, mode="val")

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    # create an instance of our ResNet model
    model = Model(num_classes=2)
    # set up loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # set up loss function
    # use weighting to tackle class imbalance
    class_counts = train[['crack', 'inactive']].sum().values
    total = len(train)
    freqs = class_counts / total
    pos_weight = (1.0 - freqs) / freqs
    criterium = nn.BCEWithLogitsLoss(pos_weight=t.Tensor(pos_weight))

    # make paths to save model and metrics
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(model_save_dir) / f"run_{now}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # create trainer instance
    trainer = Trainer(
        model=model,
        crit=criterium,
        optim=optimizer,
        train_dl=train_loader,
        val_test_dl=val_loader,
        cuda=t.cuda.is_available(),
        save_dir=run_dir,
        early_stopping_patience=early_stopping_patience
    )

    # go, go, go... call fit on trainer
    res = trainer.fit(epochs=epochs)

    # plot the results
    plt.plot(np.arange(len(res[0])), res[0], label='train loss')
    plt.plot(np.arange(len(res[1])), res[1], label='val loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('losses.png')


if __name__ == "__main__":
    run_training()