import torch as t
import numpy as np
import os
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        os.makedirs("checkpoints", exist_ok=True)
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        self._optim.zero_grad()
        output = self._model(x)
        loss = self._crit(output, y.float())
        loss.backward()
        self._optim.step()
        return loss.item()
        
        
    
    def val_test_step(self, x, y):      
        # predict
        output = self._model(x)
        # propagate through the network and calculate the loss and predictions
        loss = self._crit(output, y.float())
        # return the loss and the predictions
        # return loss value and detach output tensor from autograd
        return loss.item(), output.detach()
    
        
    def train_epoch(self):
        # set training mode
        self._model.train()
        running_loss = 0
        n_batches = 0
        # iterate through the training set
        progress = tqdm(self._train_dl, desc="Train", leave=False)
        for xb, yb in progress:
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
            # perform a training step
            loss = self.train_step(xb, yb)
            running_loss += loss
            n_batches += 1
            progress.set_postfix(loss=loss)

        # calculate the average loss for the epoch and return it
        avg_loss = running_loss / max(1, n_batches)
        return avg_loss
        
               
    def val_test(self):
        # set eval mode
        self._model.eval()
        running_loss = 0
        n_batches = 0

        all_preds = []
        all_labels = []
        # disable gradient computation
        with t.no_grad(): 
            # iterate through the validation set
            for xb, yb in self._val_test_dl:
                # transfer the batch to the gpu if given
                if self._cuda:
                    xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
                loss, probs = self.val_test_step(xb, yb)
                running_loss += loss
                n_batches += 1

                preds = t.sigmoid(probs)
                # use cpu
                preds = preds.cpu().numpy()
                labels = yb.cpu().numpy()

                # binarize predictions
                preds_bin = (preds > 0.5).astype(int)

                # save the predictions and the labels for each batch
                all_preds.append(preds_bin)
                all_labels.append(labels)

        # calculate the average loss
        val_loss = running_loss / max(1, n_batches)

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

        # return the loss and print the calculated metrics
        print(f"Validation Loss: {val_loss} | F1 Score - Crack: {f1_per_class[0]} | F1 Score - Inactive: {f1_per_class[1]}")
        return val_loss, f1_per_class
        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0, "Specify epochs or early stopping patience!"
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_losses, val_losses, f1_scores = [], [], []
        patience_count_crack = 0
        patience_count_inactive = 0
        best_f1 = np.array([0.0, 0.0])
        epoch = 0
        
        while True:
            # stop by epoch number
            if 0 <= epochs == epoch:
                break # training finished
            epoch += 1
            print(f"\nEpoch {epoch}")
            train_loss = self.train_epoch()
            val_loss, f1_per_class = self.val_test()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            f1_scores.append(f1_per_class)
            
            # if validation loss decreased save model checkpoint
            # if val_loss < best_val:
            #     best_val = val_loss
            #     patience_count = 0
            #     self.save_checkpoint(epoch)
            # else:
            #     patience_count += 1

            checkpoint_needed = False
            # f1 based early stopping Crack
            if f1_per_class[0] > best_f1[0]:
                best_f1[0] = f1_per_class[0]
                patience_count_crack = 0
                checkpoint_needed = True
            else:
                patience_count_crack += 1
            # early stopping
            if patience_count_crack >= self._early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs. No improvement for {self._early_stopping_patience} epochs in F1 - CRACK.")
                break



            # f1 based early stopping Crack
            if f1_per_class[1] > best_f1[1]:
                best_f1[1] = f1_per_class[1]
                patience_count_inactive = 0
                checkpoint_needed = True
            else:
                patience_count_inactive += 1
            # early stopping
            if patience_count_inactive >= self._early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs. No improvement for {self._early_stopping_patience} epochs in F1 - INACTIVE.")
                break

            
            if checkpoint_needed:
                self.save_checkpoint(epoch)

        return train_losses, val_losses