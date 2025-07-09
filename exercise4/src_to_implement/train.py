from training import run_training

if __name__ == "__main__":
    # run trainings here
    run_training(
        early_stopping_patience=7
    )

    run_training(
        early_stopping_patience=7,
        use_augmentation=False
    )

    run_training(
        early_stopping_patience=7,
        bs=64
    )

    run_training(
        early_stopping_patience=7,
        bs=16
    )