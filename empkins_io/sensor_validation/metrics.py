from typing import Any, Dict

import pingouin as pg
from sklearn.metrics import mean_absolute_error

from biopsykit.utils.array_handling import sanitize_input_1d

from empkins_io.utils._types import arr_t

METRICS_TEXT = {"mae": "**Mean Absolute Error**: {:.03f} bpm"}


def agreement_metrics(reference: arr_t, target: arr_t) -> Dict[str, Any]:
    reference = sanitize_input_1d(reference)
    target = sanitize_input_1d(target)
    result_dict = {"corr": pg.corr(reference, target), "mae": mean_absolute_error(reference, target)}

    return result_dict


def display_metrics(metric_dict: Dict[str, Any]) -> None:
    try:
        from IPython.core.display import Markdown, display  # pylint:disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError(
            "Displaying metric results failed because "
            "IPython cannot be imported. Install it via 'pip install ipython'."
        ) from e
    metric_dict = metric_dict.copy()
    if "corr" in metric_dict:
        display(Markdown("**Signal Correlation**"))
        display(metric_dict["corr"])
        metric_dict.pop("corr")
    for key, metric in metric_dict.items():
        display(Markdown(METRICS_TEXT[key].format(metric)))
