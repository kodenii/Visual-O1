#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR
# Modified by Richard Abrich from https://github.com/OpenAdaptAI/OpenAdapt

# from https://github.com/pytorch/extension-cpp/issues/71#issuecomment-1778326052


python -m pip install git+https://github.com/facebookresearch/detectron2.git

python setup.py build install
