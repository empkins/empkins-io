import biopsykit.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from biopsykit.signals.ecg.plotting import hr_plot
from biopsykit.utils.datatype_helper import HeartRateDataFrame, RPeakDataFrame
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple

__all__ = ["hr_plot_ecg_radar"]


def hr_plot_ecg_radar(hr_ecg: HeartRateDataFrame, hr_radar: HeartRateDataFrame, **kwargs):
    ax: plt.Axes = kwargs.pop("ax", None)

    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", None))
    else:
        fig = ax.get_figure()

    plot_outlier = kwargs.pop("plot_outlier", True)
    plot_radar_quality = kwargs.pop("plot_radar_quality", True)

    mean_ecg = hr_ecg["Heart_Rate"].mean()
    mean_radar = hr_radar["Heart_Rate"].mean()

    _hr_plot_ecg_radar(hr_ecg, hr_radar, plot_outlier, ax, **kwargs)
    _hr_plot_ecg_radar_add_legend(mean_ecg, mean_radar, ax, **kwargs)

    if plot_radar_quality and "Heartsound_Quality" in hr_radar.columns:
        ax_qual: plt.Axes = ax.twinx()
        qual_handles, qual_labels = _hr_plot_ecg_radar_quality(hr_radar, ax_qual)
        legend = ax.legend(
            qual_handles, qual_labels, ncol=3, title="Radar Heart Sound Quality", fontsize="small", loc="upper left"
        )
        ax.add_artist(legend)

        ax.set_zorder(ax_qual.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax_qual.margins(x=0)

    ax.margins(x=0)

    fig.tight_layout()
    return fig, ax


# def hr_plot_add_acc(ax: plt.Axes, acc_data: pd.DataFrame):


def _hr_plot_ecg_radar(
    hr_ecg: HeartRateDataFrame,
    hr_radar: HeartRateDataFrame,
    plot_outlier: bool,
    ax: plt.Axes,
    **kwargs,
):
    color_ecg = colors.fau_color("fau")
    mean_color_ecg = colors.adjust_color("fau")
    outlier_color_ecg = colors.fau_color("wiso")
    color_radar = colors.fau_color("med")
    mean_color_radar = colors.adjust_color("med")
    outlier_color_radar = colors.fau_color("phil")

    hr_plot(heart_rate=hr_ecg, ax=ax, color=color_ecg, mean_color=mean_color_ecg, **kwargs)
    hr_plot(heart_rate=hr_radar, ax=ax, color=color_radar, mean_color=mean_color_radar, **kwargs)

    handles = []
    labels = ["ECG", "Radar"]
    if plot_outlier:
        for hr, outlier_color in zip([hr_ecg, hr_radar], [outlier_color_ecg, outlier_color_radar]):
            h = _hr_plot_ecg_radar_outlier(hr, outlier_color, ax=ax)
            handles.append(h)

        if all(x is not None for x in handles):
            legend = ax.legend(handles, labels, title="Outlier", ncol=2, fontsize="small", loc="upper center")
            ax.add_artist(legend)


def _hr_plot_ecg_radar_outlier(data: RPeakDataFrame, color: str, ax: plt.Axes):
    h = None
    if "R_Peak_Outlier" in data.columns:
        outlier = data["R_Peak_Outlier"]
        outlier = data.index[np.where(outlier == 1)[0]]
        h = ax.scatter(x=outlier, y=data.loc[outlier, "Heart_Rate"], color=color, alpha=0.8, zorder=3)
    return h


def _hr_plot_ecg_radar_quality(hr_radar: HeartRateDataFrame, ax: plt.Axes):
    qual_handles = []
    qual_labels = ["Bad", "OK", "Good"]
    for (lims, color) in zip([(0.0, 1.8), (1.8, 2.5), (2.5, 100)], ["wiso", "phil", "nat"]):
        edgecolor = to_rgb(colors.fau_color(color)) + tuple([0.3])
        facecolor = to_rgb(colors.fau_color(color)) + tuple([0.05])
        mask = hr_radar["Heartsound_Quality"].between(*lims)
        mask = mask | mask.shift(1)
        h = ax.fill_between(
            hr_radar.index,
            0,
            hr_radar["Heartsound_Quality"],
            where=mask,
            interpolate=False,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )
        qual_handles.append(h)
    ax.set_ylabel("Heart Sound Quality [AU]")
    ax.set_ylim(bottom=0)

    return qual_handles, qual_labels


def _hr_plot_ecg_radar_add_legend(mean_ecg: float, mean_radar: float, ax: plt.Axes, **kwargs):
    plot_mean = kwargs.get("plot_mean", True)
    handles, labels = ax.get_legend_handles_labels()
    if plot_mean:
        handles = [tuple(handles[0:2]), tuple(handles[2:])]

    labels = ["{}; Mean: {:.2f} bpm".format(s, mean) for s, mean in zip(["ECG", "Radar"], [mean_ecg, mean_radar])]
    legend = ax.legend(
        handles,
        labels,
        title="Heart Rate",
        loc="upper right",
        numpoints=1,
        fontsize="small",
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )
    ax.add_artist(legend)
