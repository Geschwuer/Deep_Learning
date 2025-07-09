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

    from logger import Logger

    # =============== load and prepare data ======================
    # load the data from the csv file and perform a train-test-split
    df = pd.read_csv("data.csv", sep=";")
    train, val = train_test_split(df, test_size=0.2, random_state=seed)

    if use_augmentation:
        augmenter = DataAugmenter(train)
        train = augmenter.balance_dataset()
        train.to_csv("augmented_data.csv", index = False)

    # set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
    train_ds = ChallengeDataset(data=train, mode="train")
    val_ds = ChallengeDataset(data=val, mode="val")

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    # ================== set up model, optimizer and trainer ===================
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

    logger = Logger(save_dir=run_dir)
    logger._log("\nHyperparameters:")
    logger._log(f"Epoch: {epochs} | learning_rate: {lr} | batch_size: {bs} | weight_decay: {weight_decay} | early_stopping_patience: {early_stopping_patience} | use_augmentation: {use_augmentation} \n")
    logger._log("\n==============Logger metrics==============")
    # go, go, go... call fit on trainer
    res = trainer.fit(logger=logger, epochs=epochs)

    # ====================== plot results =======================
    # training and validation loss
    train_loss = res[0]
    val_loss = res[1]
    f1_crack = res[2][0][0]
    f1_inactive = res[2][0][1]

    plt.figure(figsize=(12, 6))
    plt.suptitle("Model Training", fontsize=14)
    
    # --- Subplot 1: F1 Scores ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, f1_crack, label="F1 Crack", color="blue")
    plt.plot(epochs, f1_inactive, label="F1 Inactive", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("F1 Scores")
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # --- Subplot 2: Loss ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label="Training Data", color="blue")
    plt.plot(epochs, val_loss, label="Validation Data", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(run_dir / "f1_and_loss_plot.png")
    plt.close()

    # class distribution of training dataset
    train["class_combo"] = train.apply(lambda row: f"{row['crack']}_{row['inactive']}", axis=1)
    sns.countplot(x="class_combo", data=train)
    plt.title("class distribution after augmentation")
    plt.xlabel("crack | inactive")
    plt.ylabel("number of samples")
    plt.savefig(run_dir / 'training_data_distribution.png')
    plt.close()