from axon.config import using_config
from axon.config import no_grad
from axon.config import test_mode
from axon.config import Config

from axon.core import Function, Variable, Parameter, as_array, as_variable
from axon.layers import Layer
from axon.models import Model
from axon.datasets import Dataset
from axon.dataloaders import DataLoader
from axon.dataloaders import SeqDataLoader

import axon.core
import axon.datasets
import axon.dataloaders
import axon.optimizers
import axon.functions
import axon.layers
import axon.utils
import axon.transforms
import axon.io

Variable.__add__ = axon.functions.add
Variable.__radd__ = axon.functions.add
Variable.__mul__ = axon.functions.mul
Variable.__rmul__ = axon.functions.mul
Variable.__neg__ = axon.functions.neg
Variable.__sub__ = axon.functions.sub
Variable.__rsub__ = axon.functions.rsub
Variable.__truediv__ = axon.functions.div
Variable.__rtruediv__ = axon.functions.rdiv
Variable.__pow__ = axon.functions.pow
Variable.__getitem__ = axon.functions.get_item

Variable.matmul = axon.functions.matmul
Variable.dot = axon.functions.matmul
Variable.max = axon.functions.max
Variable.min = axon.functions.min

__version__ = '0.0.13'