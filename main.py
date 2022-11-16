# Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2021 David W. Romero & Robert-Jan Bruintjes
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Code adapted from https://github.com/rjbruin/flexconv -- MIT License


import datetime
import os
import random

import hydra
import numpy as np
import torch

# Loggers and config
import wandb
from hydra import utils
from omegaconf import OmegaConf

# project
import partial_equiv.general as gral
import tester
import trainer
from dataset_constructor import construct_dataloaders
from model_constructor import construct_model


@hydra.main(config_path="cfg", config_name="config.yaml")
def main(cfg: OmegaConf):
    # We want to add fields to cfg so need to call OmegaConf.set_struct
    OmegaConf.set_struct(cfg, False)
    print("Input arguments:")
    print(OmegaConf.to_yaml(cfg))

    # Set visible devices for run
    if cfg.cuda_visible_devices[0] == -1:
        # If it's -1, all devices are used.
        pass
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in cfg.cuda_visible_devices)

    # Verify if the current arguments are compatible
    verify_arguments(cfg)

    # Set the seed
    set_manual_seed(cfg.seed)

    # Initialize weight and bias
    if not cfg.train or cfg.debug:
        os.environ["WANDB_MODE"] = "dryrun"
        os.environ["HYDRA_FULL_ERROR"] = "1"

    wandb.init(
        project=cfg.wandb.project,
        config=gral.utils.flatten_configdict(cfg),
        entity="dwromero",
    )

    # Construct the model
    model = construct_model(cfg)
    # Send model to GPU if available, otherwise to CPU
    # Check if multi-GPU available and if so, use the available GPU's
    print("GPU's available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    if (cfg.device == "cuda") and torch.cuda.is_available():
        cfg.device = "cuda:0"
    else:
        cfg.device = "cpu"
    model.to(cfg.device)

    # Construct dataloaders ( Dataloaders -> Dataloader["train", "validation", "test"] )
    dataloaders = construct_dataloaders(cfg)

    # Training
    if cfg.pretrained:  # TODO: Make general
        # Load model state dict
        path = utils.get_original_cwd()
        path = os.path.join(path, "saved/final_model.pt")
        model.load_state_dict(
            torch.load(path, map_location=cfg.device)["model"],
            strict=True,
        )

    # Train the model
    if cfg.train.do:
        # Print arguments (Sanity check)
        print("Modified arguments:")
        print(OmegaConf.to_yaml(cfg))
        print(datetime.datetime.now())

        trainer.train(model, dataloaders, cfg)

    # Test the model
    tester.test(model, dataloaders["test"], cfg)


def verify_arguments(
    cfg: OmegaConf,
):
    if cfg.conv.partial_equiv and cfg.base_group.sampling_method == "random" and not cfg.base_group.sample_per_layer:
        raise ValueError(
            "if cfg.conv.partial_equiv == True and cfg.base_group.sampling_method == random, "
            "cfg.base_group.sample_per_layer must be True."
            f"current values: [cfg.conv.partial_equiv={cfg.conv.partial_equiv}, "
            f"cfg.base_group.sampling_method={cfg.base_group.sampling_method}, "
            f"cfg.base_group.sample_per_layer={cfg.base_group.sample_per_layer}]"
        )

    if cfg.base_group.sampling_method == "deterministic":
        if cfg.base_group.sample_per_layer:
            raise ValueError(
                "if cfg.base_group.sampling_method == deterministic, config.base_group.sample_per_layer must be False."
                f"current values: [cfg.base_group.sampling_method={cfg.base_group.sampling_method}, "
                f"cfg.base_group.sample_per_layer={cfg.base_group.sample_per_layer}]"
            )
        if cfg.base_group.sample_per_batch_element:
            raise ValueError(
                "if cfg.base_group.sampling_method == deterministic, config.base_group.sample_per_batch_element must be False."
                f"current values: [cfg.base_group.sampling_method={cfg.base_group.sampling_method}, "
                f"cfg.base_group.sample_per_batch_element={cfg.base_group.sample_per_batch_element}]"
            )


def set_manual_seed(
    seed: int,
):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    main()
