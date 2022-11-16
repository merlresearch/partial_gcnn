# Copyright (C) 2021 David W. Romero & Robert-Jan Bruintjes
# SPDX-License-Identifier: MIT
#
# Code taken from https://github.com/rjbruin/flexconv -- MIT License


import pandas as pd
from omegaconf import OmegaConf


def flatten_configdict(
    cfg: OmegaConf,
    sep: str = ".",
):
    cfgdict = OmegaConf.to_container(cfg)
    cfgdict = pd.json_normalize(cfgdict, sep=sep)

    return cfgdict.to_dict(orient="records")[0]
