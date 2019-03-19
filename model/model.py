import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from .hourglass import *
from .smallnet import *

from .keypoint_prediction import *