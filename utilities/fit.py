import torch
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor
from typing import Callable
from torch.optim import Optimizer
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader


def train_model(
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Callable[[Tensor, Tensor], Tensor],
        dataloader: DataLoader,
        current_epoch: list[int, int],
        device: str,
        acc_func: Callable[[Tensor, Tensor], int],
        scaler: GradScaler
) -> [float, float]:
    model.train()
    running_loss = 0
    number_samples = 0
    accurate_samples = 0
    pbar = tqdm(total=len(dataloader), desc='Training: epoch {}/{}'.format(current_epoch[0], current_epoch[1]))
    for batch_idx, (ref, qry, ref_r, qry_r, label) in enumerate(dataloader):
        # transfer data to cuda
        ref, qry, ref_r, qry_r, label = ref.to(device), qry.to(device), ref_r.to(device), qry_r.to(
            device), label.float().to(device)
        if scaler:
            with autocast():
                output = model(ref, qry, ref_r, qry_r)
                output = torch.squeeze(output, 1)
                loss = criterion(output, label)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(ref, qry, ref_r, qry_r)
            output = torch.squeeze(output, 1)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # record change of loss
        number_samples += len(ref)
        running_loss += loss.item() * len(ref)
        accurate_samples += acc_func(output, label)
        pbar.set_postfix(
            current_loss='{:.3f}'.format(running_loss / number_samples),
            current_acc='{:.2%}'.format(accurate_samples / number_samples)
        )
        pbar.update(1)
    pbar.close()
    print('\n')
    return [running_loss / number_samples, accurate_samples / number_samples]


@torch.no_grad()
def eval_model(
        model: nn.Module,
        criterion: Callable[[Tensor, Tensor], Tensor],
        dataloader: DataLoader,
        current_epoch: list[int, int],
        device: str,
        acc_func: Callable[[Tensor, Tensor], int],
        scaler: GradScaler
) -> [float, float]:
    model.eval()
    running_loss = 0
    number_samples = 0
    accurate_samples = 0
    pbar = tqdm(total=len(dataloader), desc='Evaluation: epoch {}/{}'.format(current_epoch[0], current_epoch[1]))
    for batch_idx, (ref, qry, ref_r, qry_r, label) in enumerate(dataloader):
        # transfer data to cuda
        ref, qry, ref_r, qry_r, label = ref.to(device), qry.to(device), ref_r.to(device), qry_r.to(
            device), label.float().to(device)
        if scaler:
            with autocast():
                output = model(ref, qry, ref_r, qry_r)
                output = torch.squeeze(output, 1)
                loss = criterion(output, label)
        else:
            output = model(ref, qry, ref_r, qry_r)
            output = torch.squeeze(output, 1)
            loss = criterion(output, label)
        number_samples += len(ref)
        running_loss += loss.item() * len(ref)
        accurate_samples += acc_func(output, label)
        pbar.set_postfix(
            current_loss='{:.3f}'.format(running_loss / number_samples),
            current_acc='{:.2%}'.format(accurate_samples / number_samples)
        )
        pbar.update(1)
    pbar.close()
    print('\n')
    return [running_loss / number_samples, accurate_samples / number_samples]
