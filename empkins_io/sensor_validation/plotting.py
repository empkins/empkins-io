from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.ticker as mticks
import numpy as np
import pandas as pd

import seaborn as sns
import pingouin as pg

import biopsykit.colors as colors
from biopsykit.utils.array_handling import sanitize_input_1d

from empkins_io.utils._types import arr_t

__all__ = ["plot_sensor_agreement"]


def plot_sensor_agreement(reference: arr_t, target: arr_t, **kwargs):
    fig = plt.figure(constrained_layout=False, figsize=kwargs.get("figsize", (10, 4)))
    spec = gs.GridSpec(ncols=2, nrows=1, width_ratios=[2, 3])
    axs = {}

    reference_name = kwargs.get("reference_name", None)
    target_name = kwargs.get("target_name", None)

    reference = _sanitize_input_series(reference, reference_name)
    target = _sanitize_input_series(target, target_name)

    spec_within = gs.GridSpecFromSubplotSpec(5, 5, subplot_spec=spec[0, 0], wspace=0.025, hspace=0.05)
    axs["scatter"] = fig.add_subplot(spec_within[1:5, 0:4])
    axs["marg_x"] = fig.add_subplot(spec_within[0, 0:4])
    axs["marg_y"] = fig.add_subplot(spec_within[1:5, 4])
    axs["scatter"].get_shared_x_axes().join(axs["scatter"], axs["marg_x"])
    axs["scatter"].get_shared_y_axes().join(axs["scatter"], axs["marg_y"])

    axs["bland"] = fig.add_subplot(spec[0, 1])

    _scatter_plot(reference, target, axs, **kwargs)
    _bland_altman(reference, target, axs)

    fig.tight_layout()


def _scatter_plot(reference: arr_t, target: arr_t, axs: Dict[str, plt.Axes], **kwargs):
    axlims = kwargs.get("axlims", False)
    sns.regplot(
        x=reference,
        y=target,
        ax=axs["scatter"],
        scatter_kws={
            "alpha": 0.5,
        },
    )
    if axlims:
        llim = np.min([np.min(reference), np.min(target)])
        rlim = np.max([np.max(reference), np.max(target)])
        axs["scatter"].set_xlim([llim, rlim])
        axs["scatter"].set_ylim([llim, rlim])

    sns.histplot(x=reference, kde=True, bins=int(len(reference) / 10), ax=axs["marg_x"])
    sns.histplot(y=target, kde=True, bins=int(len(target) / 10), ax=axs["marg_y"])

    axs["scatter"].tick_params(axis="both", which="both", left=True, bottom=True)
    axs["scatter"].tick_params(axis="x", labelrotation=30)
    axs["scatter"].xaxis.set_major_locator(mticks.MultipleLocator(5))
    axs["scatter"].xaxis.set_minor_locator(mticks.MultipleLocator(1))
    axs["scatter"].yaxis.set_major_locator(mticks.MultipleLocator(5))
    axs["scatter"].yaxis.set_minor_locator(mticks.MultipleLocator(1))
    axs["marg_x"].axis("off")
    axs["marg_y"].axis("off")


def _bland_altman(reference: arr_t, target: arr_t, axs: Dict[str, plt.Axes]):
    pg.plot_blandaltman(
        x=reference, y=target, scatter_kws={"color": colors.fau_color("fau"), "alpha": 0.6}, ax=axs["bland"]
    )


def _sanitize_input_series(data: arr_t, data_name: Optional[str] = None) -> pd.Series:
    data_cpy = sanitize_input_1d(data)
    if data_name is None:
        if isinstance(data, pd.Series):
            data_name = data.name
        elif isinstance(data, pd.DataFrame):
            data_name = data.columns[0]
    return pd.Series(data_cpy, name=data_name)
