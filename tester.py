# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 David W. Romero & Robert-Jan Bruintjes
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Code adapted from https://github.com/rjbruin/flexconv -- MIT License


import torch
import wandb
from tqdm import tqdm

# project
from globals import IMG_DATASETS


def test(model, test_loader, cfg, **kwargs):

    # Define criterion and training function
    if cfg.dataset in IMG_DATASETS:
        test_function = classification_test
    else:
        raise NotImplementedError(f"No test function found for dataset {cfg.dataset}.")

    return test_function(model, test_loader, cfg)


def classification_test(model, test_loader, cfg):
    # send model to device
    device = cfg.device

    model.eval()
    model.to(device)

    # Summarize results
    correct = 0
    total = 0

    with torch.no_grad():
        # Iterate through data
        for data in tqdm(test_loader, desc="Test"):

            inputs, labels = data
            # inputs = torch.rot90(inputs, k=3, dims=(-1,-2))
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print results
    test_acc = correct / total
    wandb.run.summary["best_test_accuracy"] = test_acc
    print(f"Accuracy of the network on the {total} test samples: {(100 * test_acc)}")

    return test_acc
