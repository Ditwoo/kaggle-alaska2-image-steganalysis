# installed
from catalyst.dl import registry
from catalyst.dl import SupervisedRunner as Runner
# local
from .experiment import Experiment
from .metrics import WeightedAUC
from .schedulers import CosineAnnealingWithRestartsLR
from .models import (
    BinaryEfficientNet,
    BinaryDensenet,
)


registry.Callback(WeightedAUC)

registry.Scheduler(CosineAnnealingWithRestartsLR)

registry.Model(BinaryEfficientNet)
registry.Model(BinaryDensenet)
