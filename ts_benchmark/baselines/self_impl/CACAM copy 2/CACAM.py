import time
import copy
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch import optim

from ts_benchmark.baselines.self_impl.CACAM.models.CACAM_model import Basic_CACAM
from ts_benchmark.baselines.utils import anomaly_detection_data_provider
from ts_benchmark.utils.device import get_available_device

DEFAULT_CACAM_BASED_HYPER_PARAMS = {
    "seq_len": 100,
    "d_model": 128,
    "n_heads": 4,
    "dropout": 0.1,
    "causal_method": "correlation",
    "causal_max_lag": 3,
    "causal_pc_alpha": 0.05,
    "train_epochs": 10,
    "batch_size": 128,
    "optim": "adam",
    "learning_rate": 1e-4,
    "lradj": "type1",
    "patience": 3,
    "pct_start": 0.3,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
}


def _adjust_learning_rate(optimizer, epoch, train_configs, verbose=True, **other_args):
    if train_configs.lradj == "type1":
        lr_adjust = {
            epoch: train_configs.learning_rate * (0.5 ** ((epoch - 1) // 1))
        }
    elif train_configs.lradj == "type2":
        lr_adjust = {
            2: 5e-5,
            4: 1e-5,
            6: 5e-6,
            8: 1e-6,
            10: 5e-7,
            15: 1e-7,
            20: 5e-8,
        }
    elif train_configs.lradj == "type3":
        lr_adjust = {
            epoch: train_configs.learning_rate
            if epoch < 3
            else train_configs.learning_rate * (0.9 ** ((epoch - 3) // 1))
        }
    elif train_configs.lradj == "cosine":
        lr_adjust = {
            epoch: train_configs.learning_rate
            / 2
            * (1 + math.cos(epoch / train_configs.train_epochs * math.pi))
        }
    elif train_configs.lradj == "1cycle":
        scheduler = other_args["scheduler"]
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    else:
        lr_adjust = {}

    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        if verbose:
            print(f"Updating learning rate to {lr}")


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.check_point = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.check_point = copy.deepcopy(model.state_dict())
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        self.val_loss_min = val_loss

class CACAMConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_CACAM_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

class CACAM:
    def __init__(self, **kwargs):
        super(CACAM, self).__init__()
        self.config = CACAMConfig(**kwargs)
        self.scaler = StandardScaler()
        self.model_name = "CACAM"
        self.device = get_available_device()
        self.model = Basic_CACAM(self.config).to(self.device)
        self.criterion = nn.MSELoss()

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def detect_validate(self, valid_data_loader):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (input, _) in enumerate(valid_data_loader):
                input = input.float().to(self.device)
                reconstructed, _ = self.model(input)
                loss = self.criterion(reconstructed, input)
                total_loss.append(loss.item())
        self.model.train()
        return np.average(total_loss)

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame = None):
        from ts_benchmark.baselines.utils import train_val_split
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        self.scaler.fit(train_data_value.values)
        
        train_data_scaled = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index
        )
        
        self.train_data_loader = anomaly_detection_data_provider(
            train_data_scaled,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="train",
        )
        
        if valid_data is not None:
            if isinstance(valid_data, tuple):
                valid_data = valid_data[0] # handle split_before tuple return if applicable
                
            if len(valid_data.shape) > 0 and valid_data.shape[0] > 0:
                valid_data_scaled = pd.DataFrame(
                    self.scaler.transform(valid_data.values),
                    columns=valid_data.columns,
                    index=valid_data.index
                )
                self.valid_data_loader = anomaly_detection_data_provider(
                    valid_data_scaled,
                    batch_size=self.config.batch_size,
                    win_size=self.config.seq_len,
                    step=1,
                    mode="val",
                )
            else:
                self.valid_data_loader = self.train_data_loader
        else:
            self.valid_data_loader = self.train_data_loader

        self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)
        train_steps = len(self.train_data_loader)

        if self.config.optim == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optim == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        if self.config.lradj == "1cycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                steps_per_epoch=train_steps,
                pct_start=self.config.pct_start,
                epochs=self.config.train_epochs,
                max_lr=self.config.learning_rate
            )

        time_now = time.time()
        for epoch in range(self.config.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            self.model.train()

            for i, (input, target) in enumerate(self.train_data_loader):
                iter_count += 1
                self.optimizer.zero_grad()
                input = input.float().to(self.device)
                
                reconstructed, _ = self.model(input)
                loss = self.criterion(reconstructed, input)
                
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.config.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.optimizer.step()

                if self.config.lradj == '1cycle':
                    _adjust_learning_rate(self.optimizer, epoch + 1, self.config, verbose=False, scheduler=scheduler)
                    scheduler.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss_avg = np.average(train_loss)
            vali_loss = self.detect_validate(self.valid_data_loader)
            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss_avg:.7f} Vali Loss: {vali_loss:.7f}")

            self.early_stopping(vali_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            if self.config.lradj != "1cycle":
                _adjust_learning_rate(self.optimizer, epoch + 1, self.config)

    def detect_score(self, test: pd.DataFrame) -> np.ndarray:
        test_scaled = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        self.model.load_state_dict(self.early_stopping.check_point)
        
        test_loader = anomaly_detection_data_provider(
            test_scaled,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="thre",
        )

        self.model.eval()
        attens_energy = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                reconstructed, _ = self.model(batch_x)
                
                # Anomaly score based on reconstruction error
                # Calculate MSE per point [B, T, C]
                error = (reconstructed - batch_x) ** 2
                score = error.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0) # [nb x t x c]
        attens_energy = attens_energy.reshape(-1, attens_energy.shape[-1]) # [nb*t x c]
        test_energy = np.mean(attens_energy, axis=-1) # [nb*t]
        
        # Return energy as score. ts_benchmark typically requires return score, score or similar
        return test_energy, test_energy

    def detect_label(self, test: pd.DataFrame) -> np.ndarray:
        # Implement a naive thresholding based on training data reconstruction error
        # Normally ts_benchmark handles thresholding itself with detect_score and its metrics.
        # But we implement this to satisfy interface.
        self.model.load_state_dict(self.early_stopping.check_point)
        self.model.eval()
        
        train_energy = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.train_data_loader):
                batch_x = batch_x.float().to(self.device)
                reconstructed, _ = self.model(batch_x)
                error = (reconstructed - batch_x) ** 2
                train_energy.append(error.detach().cpu().numpy())
                
        train_energy = np.concatenate(train_energy, axis=0)
        train_energy = train_energy.reshape(-1, train_energy.shape[-1])
        train_energy = np.mean(train_energy, axis=-1)
        
        test_energy, _ = self.detect_score(test)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        
        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]
            
        preds = {}
        for ratio in self.config.anomaly_ratio:
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)
            
        return preds, test_energy
