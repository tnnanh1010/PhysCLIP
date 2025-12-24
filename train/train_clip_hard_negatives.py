"""
CLIP Fine-tuning for Astronomy Images - Hard Negative Mining Version
Uses hard negative mining for improved contrastive learning
"""

import os
import json
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import (
    VisionTextDualEncoderModel,
    AutoTokenizer,
    CLIPImageProcessor,
    get_cosine_schedule_with_warmup,
)
from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb
