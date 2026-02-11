#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

import torch
from fairseq import checkpoint_utils, options
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import utils as distributed_utils

from fairseq_cli.train import *


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    state = checkpoint_utils.load_checkpoint_to_cpu('unit_mbart.pt')
    assert state.get("cfg") is not None
    config_dict = vars(state["cfg"]["task"])
    args = argparse.Namespace(
        **{k: config_dict[k] if k in config_dict.keys() else v for k, v in vars(args).items()}
    )

    import yaml
    with open('conf/utut_no_encoder_loss.yaml') as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)
    args = argparse.Namespace(
        **{k: config_dict[k] if k in config_dict.keys() else v for k, v in vars(args).items()}
    )

    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(
            f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}"
        )

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

if __name__ == "__main__":
    cli_main()
