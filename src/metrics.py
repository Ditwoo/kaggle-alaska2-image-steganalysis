import torch
import numpy as np
from sklearn import metrics
from catalyst.dl import Callback, CallbackOrder, State


import warnings
warnings.filterwarnings('ignore') 


def _alaska_weighted_auc(y_true: np.ndarray, y_valid: np.ndarray) -> float:
    """
    Source:
        https://www.kaggle.com/anokas/weighted-auc-metric-updated
    """
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights =        [       2,   1]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)
    
    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])
    
    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)
    
    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)

        if mask.sum() == 0:
            continue

        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min # normalize such that curve starts at y=0
        score = metrics.auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric

    return competition_metric / normalization



class WeightedAUC(Callback):
    def __init__(self, 
                 name: str = "wauc",
                 input_key: str = "targets",
                 output_key: str = "logits"):
        super().__init__(CallbackOrder.Metric)
        self.name = name
        self.inp = input_key
        self.outp = output_key
        self.target_container = None
        self.preds_container = None

    def on_loader_start(self, state: State) -> None:
        self.target_container = []
        self.preds_container = []

    def on_batch_end(self, state: State) -> None:
        target = state.input[self.inp].detach().cpu().numpy().astype(int).flatten()
        self.target_container.append(target)

        pred = torch.sigmoid(state.output[self.outp].detach()).cpu().numpy().flatten()
        self.preds_container.append(pred)

        state.batch_metrics[f"batch_{self.name}"] = _alaska_weighted_auc(target, pred)

    def on_loader_end(self, state: State) -> None:
        score = _alaska_weighted_auc(
            np.concatenate(self.target_container),
            np.concatenate(self.preds_container)
        )
        state.loader_metrics[self.name] = score
        # free memory
        self.target_container = None
        self.preds_container = None