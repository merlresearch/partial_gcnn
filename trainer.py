# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 David W. Romero & Robert-Jan Bruintjes
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Code adapted from https://github.com/rjbruin/flexconv -- MIT License

import copy
import datetime
import os

# typing
from typing import Dict

# torch
import torch

# logger
import wandb
from hydra import utils
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import optim
import partial_equiv.general as gral
import partial_equiv.partial_gconv as partial_gconv

# project
import tester
from globals import IMG_DATASETS
from partial_equiv import ck


def train(
    model: torch.nn.Module,
    dataloaders: Dict[str, DataLoader],
    cfg: OmegaConf,
):

    # Define criterion and training function
    if cfg.dataset in IMG_DATASETS:
        criterion = torch.nn.CrossEntropyLoss().to(cfg.device)
        train_function = classification_train
    else:
        raise NotImplementedError(f"No training criterion and training function found for dataset {cfg.dataset}.")

    # Define optimizer and scheduler
    optimizer = optim.construct_optimizer(model, cfg)
    lr_scheduler = optim.construct_scheduler(optimizer, cfg)

    # Train model
    train_function(
        model,
        criterion,
        optimizer,
        dataloaders,
        lr_scheduler,
        cfg,
    )

    # Save the final model
    save_model_to_wandb(model, optimizer, lr_scheduler, name="final_model")

    if cfg.debug:
        # If running on debug mode, also save the model locally.
        torch.save(model.state_dict(), os.path.join(utils.get_original_cwd(), "saved/model.pt"))

    return


def classification_train(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloaders: Dict[str, DataLoader],
    lr_scheduler,
    cfg: OmegaConf,
):
    # Construct weight_regularizer
    weight_regularizer = gral.nn.loss.LnLoss(weight_loss=cfg.train.weight_decay, norm_type=2)

    # Construct decay_regularizer
    mono_decay_loss = gral.nn.loss.MonotonicPartialEquivarianceLoss(
        weight_loss=cfg.train.monotonic_decay_loss,
    )

    # Training parameters
    epochs = cfg.train.epochs
    device = cfg.device

    # Save best performing weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 999

    # Counter for epochs without improvement
    epochs_no_improvement = 0
    max_epochs_no_improvement = 100

    # iterate over epochs
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("-" * 30)
        # Print current learning rate
        for param_group in optimizer.param_groups:
            print("Learning Rate: {}".format(param_group["lr"]))
        print("-" * 30)
        # log learning_rate of the epoch
        wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=epoch + 1)

        # Each epoch consist of training and validation
        for phase in ["train", "validation"]:

            if phase == "train":
                model.train()
            else:
                model.eval()

            # Accumulate accuracy and loss
            running_loss = 0
            running_corrects = 0
            total = 0

            # iterate over data
            for data in tqdm(dataloaders[phase], desc=f"Epoch {epoch} / {phase}"):

                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                train = phase == "train"

                with torch.set_grad_enabled(train):
                    # Fwrd pass:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # Regularization:
                    if cfg.train.weight_decay > 0.0:
                        loss = loss + weight_regularizer(model)
                    if cfg.train.monotonic_decay_loss > 0.0:
                        loss = loss + mono_decay_loss(model)

                    if phase == "train":

                        # Backward pass
                        loss.backward()

                        # Gradient clip
                        if cfg.train.gradient_clip != 0.0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                cfg.train.gradient_clip,
                            )

                        # Optimizer step
                        optimizer.step()

                        # update the lr_scheduler
                        if isinstance(
                            lr_scheduler,
                            (
                                torch.optim.lr_scheduler.CosineAnnealingLR,
                                gral.lr_scheduler.LinearWarmUp_LRScheduler,
                            ),
                        ):
                            lr_scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)

            # statistics of the epoch
            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            print(datetime.datetime.now())

            # log statistics of the epoch
            wandb.log(
                {
                    "accuracy" + "_" + phase: epoch_acc,
                    "loss" + "_" + phase: epoch_loss,
                },
                step=epoch + 1,
            )

            # If better validation accuracy, replace best weights and compute the test performance
            if phase == "validation" and epoch_acc >= best_acc:

                # Updates to the weights will not happen if the accuracy is equal but loss does not diminish
                if (epoch_acc == best_acc) and (epoch_loss > best_loss):
                    pass
                else:
                    best_acc = epoch_acc
                    best_loss = epoch_loss

                    best_model_wts = copy.deepcopy(model.state_dict())
                    save_model_to_wandb(model, optimizer, lr_scheduler, epoch=epoch + 1)

                    # Log best results so far and the weights of the model.
                    wandb.run.summary["best_val_accuracy"] = best_acc
                    wandb.run.summary["best_val_loss"] = best_loss

                    # Clean CUDA Memory
                    del inputs, outputs, labels
                    torch.cuda.empty_cache()

                    # # Perform test and log results
                    # if cfg.dataset in ["rotMNIST", "CIFAR10", "CIFAR10", "CIFAR100", "PCam"]:
                    #     test_acc = tester.test(model, dataloaders["test"], cfg)
                    # else:
                    #     test_acc = best_acc
                    # wandb.run.summary["best_test_accuracy"] = test_acc
                    # wandb.log(
                    #     {"accuracy_test": test_acc},
                    #     step=epoch + 1,
                    # )

                    # Reset counter of epochs without progress
                    epochs_no_improvement = 0

            elif phase == "validation" and epoch_acc < best_acc:
                # Otherwise, increase counter
                epochs_no_improvement += 1

            if phase == "validation" and cfg.conv.partial_equiv:
                # Log to wandb and print
                log_and_print_probabilities(model, step=epoch + 1)

            # Log omega_0
            if cfg.kernel.type == "SIREN" and cfg.kernel.learn_omega0:
                log_and_print_omega_0s(model, epoch + 1, log_to_wandb=True)

            # Update scheduler
            if phase == "validation" and isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(epoch_acc)

        # Update scheduler
        if isinstance(
            lr_scheduler,
            (
                torch.optim.lr_scheduler.MultiStepLR,
                torch.optim.lr_scheduler.ExponentialLR,
            ),
        ):
            lr_scheduler.step()
        print()

        #  Check how many epochs without improvement have passed, and, if required, stop training.
        if epochs_no_improvement == max_epochs_no_improvement:
            print(f"Stopping training due to {epochs_no_improvement} epochs of no improvement in validation accuracy.")
            break

    # Report best results
    print("Best Val Acc: {:.4f}".format(best_acc))
    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Print learned w0s
    if cfg.kernel.type == "SIREN" and cfg.kernel.learn_omega0:
        log_and_print_omega_0s(model, step=-1, log_to_wandb=False)

    # Return model
    return model


def save_model_to_wandb(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    name: str = "model",
    epoch=None,
):
    filename = f"{name}.pt"
    if epoch is not None:
        filename = "checkpoint.pt"
    path = os.path.join(wandb.run.dir, filename)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "epoch": epoch,
        },
        path,
    )
    # Call wandb to save the object, syncing it directly
    wandb.save(path)


def log_and_print_probabilities(
    model: torch.nn.Module,
    step: int,
):
    probs = {}
    counter = 0
    for m in model.modules():
        if isinstance(m, partial_gconv.GroupConv):
            # get
            prob = m.probs.detach().cpu()
            # print
            print(prob)
            # add to probs dict
            probs[str(counter)] = prob
            # increase counter
            counter += 1
    # log probs:
    wandb.log({"probs": probs}, step=step)


def log_and_print_omega_0s(
    model: torch.nn.Module,
    step: int,
    log_to_wandb: bool,
):
    w0s = {}
    counter = 0
    for m in model.modules():
        if isinstance(
            m,
            (
                ck.siren.SIRENLayer1d,
                ck.siren.SIRENLayer2d,
                ck.siren.SIRENLayer3d,
                ck.siren.SIRENLayerNd,
            ),
        ):
            w0 = m.omega_0.detach().cpu().item()
            # print
            print(w0)
            # add to probs dict
            w0s[str(counter)] = w0
            # increase counter
            counter += 1
    if log_to_wandb:
        # log probs:
        wandb.log({"w0s": w0s}, step=step)
