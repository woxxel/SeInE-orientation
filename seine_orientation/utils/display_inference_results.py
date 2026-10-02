import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from turnover_dynamics.place_selectivity_inference.utils import gauss_smooth

from turnover_dynamics.orientation_selectivity_inference.BayesModel import (
    HierarchicalBayesInference,
)
from turnover_dynamics.orientation_selectivity_inference.utils.utils_model import (
    gabor_filter,
    gabor_response,
)
from turnover_dynamics.orientation_selectivity_inference.hierarchical_bayes_inference.NestedSamplingMethods import (
    get_posteriors,
    plot_results,
)


def plot_3D(ax, rate, x, y, clims=None, cmap=plt.cm.viridis, **options):
    norm = mcolors.Normalize(vmin=0, vmax=5)
    """firingmap!"""
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, rate, facecolors=cmap(norm(rate)), **options)
    ax.set_zlim(clims)


def plot_gabor_filter(ax, X, Y, params, margin=5):

    steps = X.shape[0]
    G, significant_region = gabor_filter(X, Y, **params, significance_threshold=0.001)
    sig = np.where(significant_region)

    y_min, y_max = np.maximum(0, np.min(sig[0]) - margin), np.minimum(
        np.max(sig[0]) + margin, steps - 1
    )
    x_min, x_max = np.maximum(0, np.min(sig[1]) - margin), np.minimum(
        np.max(sig[1]) + margin, steps - 1
    )

    y_lims = Y[[y_min, y_max], 0]
    x_lims = X[0, [x_min, x_max]]

    G_plot = np.copy(G)
    G_plot[np.abs(G) < G.max() * 0.01] = np.nan

    # ax_gabor = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(X, Y, G_plot, cmap="coolwarm", alpha=0.8)
    ax.contour(X, Y, G_plot, zdir="z", offset=-1, cmap="coolwarm")
    ax.contour(X, Y, G_plot, zdir="x", offset=x_lims[0], cmap="coolwarm")
    ax.contour(X, Y, G_plot, zdir="y", offset=y_lims[1], cmap="coolwarm")
    plt.setp(
        ax,
        xlim=x_lims,
        ylim=y_lims,
        zlim=[-1.0, 1.0],
        xlabel="X",
        ylabel="Y",
        zlabel="Gabor filter amplitude",
    )
    ax.zaxis.set_major_locator(plt.MaxNLocator(3, symmetric=True))


def plot_overview(
    results_all,
    neuron,
    data,
    dwelltime,
    measure_points,
    model=None,
    mode="mean",
    sigma=(0.5, 1.0),
):
    if neuron not in results_all:
        results = results_all
    else:
        results = results_all[neuron]

    if model is None:
        model, model_descr = test_winning_model(results)
        # print(model)
    else:
        model_descr = ""

    hbm = HierarchicalBayesInference()
    hbm.set_gratings(
        measure_points,
        [np.deg2rad(140), np.deg2rad(114)],
        FoV_steps=201,
    )
    # if event_counts.ndim == 4:
    #     data = event_counts[..., neuron].sum(axis=0)
    #     dwelltime_sum = dwelltime.sum(axis=0)
    # else:
    # data = event_counts[..., neuron]
    # dwelltime_sum = np.copy(dwelltime)
    hbm.prepare_data(
        data,
        dwelltime,
        iter_dims=False,
        # dimension_names=unique_values.keys(), iter_dims=[False, False, False],
        # collapse_repeats=True
    )
    hbm.set_priors(coding=model)
    if mode == "mean":
        posteriors = get_posteriors(hbm, results[model])
        params = {
            key: float(posteriors[key]["mean"]) for key in hbm.parameter_names_all
        }
    else:
        params = hbm.get_params_from_p(results[model].samples[-1])
    rate_response_model = hbm.get_model_rate_response(params, coding=model)

    fig = plt.figure(figsize=(16, 6), layout="constrained")
    fig_posteriors, fig_results = fig.subfigures(1, 2, width_ratios=[0.2, 0.8])

    plot_results(hbm, results[model], fig=fig_posteriors)

    gs = fig_results.add_gridspec(2, 5, wspace=0.1, hspace=0.15)
    gs_small = fig_results.add_gridspec(4, 10, wspace=0.1)
    if model in ["simple", "complex"]:
        ax_gabor = fig_results.add_subplot(gs_small[1:, :3], projection="3d")
        plot_gabor_filter(ax_gabor, hbm.X_FoV, hbm.Y_FoV, params)
        idx_offset = 2
    else:
        # gs = fig_results.add_gridspec(2, 5, wspace=0.05,hspace=0.15)
        # gs_small = fig_results.add_gridspec(4, 10)
        idx_offset = 2

    ax_models = fig_results.add_subplot(gs_small[0, 0])
    xticks = []
    for i, (key, res) in enumerate(results.items()):
        ax_models.errorbar(
            i,
            res.logz[-1],
            yerr=res.logzerr[-1],
            color="tab:green" if key == model else "k",
            linestyle="none",
            marker="o",
            elinewidth=2,
        )
        if key == model:
            print(
                f"Model {model} log evidence: {res.logz[-1]:.2f} ± {res.logzerr[-1]:.2f}, favored over random: {model_descr}"
            )
            ax_models.text(
                i,
                res.logz[-1] + 1,
                s=Jeffreys_to_stars.get(model_descr, ""),
                color="tab:green",
                fontsize=8,
                ha="center",
                va="bottom",
            )
        xticks.append(key)
    plt.setp(
        ax_models,
        xlim=(-0.5, len(xticks) - 0.5),
        xticks=range(len(xticks)),
        ylabel="log evidence",
    )
    ax_models.set_xticklabels(xticks, rotation=60)
    ax_models.spines[["top", "right"]].set_visible(False)
    # ax_models.ticklabel_format(axis="y", style="plain")
    ax_models.ticklabel_format(useOffset=False, axis="y", style="plain")

    ax_models.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # ax_models.ticklabel_format(axis="y", style="sci", scilimits=(-2,5))

    ax_nonlinearities = fig_results.add_subplot(gs_small[1, 3])
    xmin = -0.5
    xmax = 1.0
    x = np.linspace(xmin, xmax, 100)
    # ax_nonlinearities.axline((0, 0), slope=1, color="k", linestyle="--", alpha=1.0)
    ax_nonlinearities.plot(
        (xmin, 0),
        (params["nl_baseline"], params["nl_baseline"]),
        color="k",
        linestyle="--",
        linewidth=0.5,
    )
    ax_nonlinearities.plot(
        (0, xmax),
        (
            params["nl_baseline"],
            params["nl_baseline"] + params.get("nl_amplitude", 0.0),
        ),
        color="k",
        linestyle="--",
        linewidth=0.5,
    )
    ax_nonlinearities.plot(x, hbm.apply_nonlinearity(x, params), color="tab:green")
    ax_nonlinearities.spines[["top", "right"]].set_visible(False)
    # plt.setp(ax_nonlinearities, xlim=(xmin, xmax),ylim=(0, params["nl_baseline"]+params.get("nl_amplitude", 0.0)*1.2))
    plt.setp(
        ax_nonlinearities,
        xlim=(xmin, xmax),
        ylim=(0, np.max(hbm.apply_nonlinearity(x, params)) * 1.2),
    )
    ax_nonlinearities.set_xlabel("Gabor response (norm.)")
    ax_nonlinearities.set_ylabel("Firing rate (Hz)")

    rate_response_data = hbm.data["observed_counts"] / hbm.data["T"]

    ax_logl = fig_results.add_subplot(gs_small[0, 2])
    mu = params["nl_baseline"] + params.get("nl_amplitude", 0.0) / 2.0
    T = np.mean(hbm.data["T"])
    count_arr = np.arange(max(10, int(4 * T * mu)))

    logp_poisson = hbm.probability_of_spike_observation(
        mu, observed_counts=count_arr, T=np.mean(hbm.data["T"]), model="poisson"
    )
    logp_negbin = hbm.probability_of_spike_observation(
        mu,
        observed_counts=count_arr,
        T=np.mean(hbm.data["T"]),
        model="negative_binomial",
        **params,
    )

    ax_logl.axvline(mu, color="k", linestyle="--", linewidth=0.5)
    ax_logl.plot(
        count_arr,
        logp_poisson,
        label="Poisson log-likelihood",
        color="k",
        linestyle="--",
        linewidth=1.0,
    )
    ax_logl.plot(
        count_arr,
        logp_negbin,
        label="Negative binomial log-likelihood",
        color="tab:orange",
        linewidth=2.0,
    )
    ax_logl.set_xlabel("Spike count")
    ax_logl.set_ylabel("Log-likelihood")
    # ax_logl.legend()
    ax_logl.spines[["top", "right"]].set_visible(False)

    # clim = np.percentile(rate_response_model, [1, 99])
    # print("sigma:", (0,)+sigma)
    # print(rate_response_data.shape)
    clim = np.percentile(gauss_smooth(rate_response_data, (0,) + sigma), [1, 99])
    clim[0] = 0
    clim[1] *= 1.2
    # norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])

    for i in range(3):
        opts = {
            "x": measure_points[1],
            "y": measure_points[2],
            "clims": clim,
            "alpha": 0.8,
            "edgecolor": None,
        }
        ax = fig_results.add_subplot(gs[0, i + idx_offset], projection="3d")
        plot_3D(ax, gauss_smooth(rate_response_data[i, ...].T, sigma), **opts)
        ax.set_title(f"{measure_points[0][i]:.0f}°")

        ax = fig_results.add_subplot(gs[1, i + idx_offset], projection="3d")
        plot_3D(ax, rate_response_model[i, ...].T, **opts)

    ax = plt.subplot(gs[1, -1])
    plt.setp(ax, xlabel="Angle", ylabel="Cycles", zlabel="Firing rate (Hz)")

    ax_data = fig_results.add_subplot(gs[0, idx_offset:])
    ax_data.axis("off")
    ax_data.set_title("Firing rate response data", pad=20)

    ax_data = fig_results.add_subplot(gs[1, idx_offset:])
    ax_data.axis("off")
    ax_data.set_title("Firing rate response from model")
    fig.tight_layout()

    fig_results.subplots_adjust(left=0.05)
    # plt.tight_layout()
    plt.show()


from typing import Optional

### find most probable model for each neuron
Jeffrey_scale = {
    0: "inconclusively",
    1: "weakly",
    3: "moderately",
    5: "strongly",
}

Jeffreys_to_stars = {
    "inconclusively": "",
    "weakly": "*",
    "moderately": "**",
    "strongly": "***",
}


def compare_models(
    log_evidences, idx1, idx2, min_level=1
) -> tuple[float, Optional[str]]:
    dZ = log_evidences[idx1, 0] - log_evidences[idx2, 0]
    dZ_sigma = np.sqrt(log_evidences[idx1, 1] ** 2 + log_evidences[idx2, 1] ** 2)

    # print(f"Model {idx1} vs model {idx2}: dZ = {dZ:.2f} ± {dZ_sigma:.2f}")
    Z_thr = dZ - 2 * dZ_sigma

    win = None
    for threshold, description in sorted(Jeffrey_scale.items(), reverse=True):
        if Z_thr > max(threshold, min_level):
            # print(f"Model {idx1} is {description} favored over model {idx2} (Jeffreys scale: {Z_thr:.2f})")
            win = description
            break
    return Z_thr, win


def test_winning_model(results, print_it=False) -> tuple[str, str]:

    log_evidences = np.array(
        [[res_model.logz[-1], res_model.logzerr[-1]] for res_model in results.values()]
    )
    models = {"random": 0, "simple": 1, "complex": 2}
    model_names = {0: "random", 1: "simple", 2: "complex"}
    ## test if any gabor model is favored over random
    # dZ_simple_random, favored_simple = compare_models(
    dZ_simple_random, favored_random_simple = compare_models(
        # log_evidences, models["simple"], models["random"], min_level=1
        log_evidences,
        models["random"],
        models["simple"],
        min_level=0,
    )
    # dZ_complex_random, favored_complex = compare_models(
    dZ_complex_random, favored_random_complex = compare_models(
        # log_evidences, models["complex"], models["random"], min_level=1
        log_evidences,
        models["random"],
        models["complex"],
        min_level=0,
    )

    if favored_random_simple and favored_random_complex:
        if print_it:
            print(
                f"Random model is favored (dZ simple-random = {dZ_simple_random:.2f}, dZ complex-random = {dZ_complex_random:.2f})"
            )
        return "random", favored_random_simple
    elif favored_random_simple and not favored_random_complex:
        if print_it:
            print(
                f"Complex model is favored {favored_random_complex} with dZ = {dZ_simple_random:.2f}"
            )
        return "complex", favored_random_complex
    elif favored_random_complex and not favored_random_simple:
        if print_it:
            print(
                f"Simple model is favored {favored_random_simple} with dZ = {dZ_complex_random:.2f}"
            )
        return "simple", favored_random_simple
    else:  # not favored_random_simple and not favored_random_complex:

        dZ_simple_complex, favored_simple_complex = compare_models(
            log_evidences, models["simple"], models["complex"], min_level=0
        )
        dZ_complex_simple, favored_complex_simple = compare_models(
            log_evidences, models["complex"], models["simple"], min_level=0
        )
        if favored_simple_complex:
            if print_it:
                print(
                    f"Simple model is favored {favored_simple_complex} over complex model with dZ = {dZ_simple_complex:.2f}"
                )
            return "simple", favored_simple_complex
        elif favored_complex_simple:
            if print_it:
                print(
                    f"Complex model is favored {favored_complex_simple} over simple model with dZ = {dZ_complex_simple:.2f}"
                )
            return "complex", favored_complex_simple
        else:
            # if both models are favored over random, but not favored 
            # against each other, we can still say that the simple 
            # model is favored over random, even if we cannot say that 
            # it is favored over the complex model
            if print_it:
                print(
                    f"No clear winner between simple and complex model (dZ = {dZ_simple_complex:.2f})"
                )
            return (
                model_names[np.argmax(log_evidences[1:, 0], axis=0) + 1],
                "",
            )  

    # return "simple"

    if favored_simple and not favored_complex:
        if print_it:
            print(
                f"Simple model is favored {favored_simple} over random model with dZ = {dZ_simple_random:.2f}"
            )
        return "simple"
    elif favored_complex and not favored_simple:
        if print_it:
            print(
                f"Complex model is favored {favored_complex} over random model with dZ = {dZ_complex_random:.2f}"
            )
        return "complex"
    elif favored_simple and favored_complex:
        if print_it:
            print(
                f"Both simple and complex model are favored over random model (dZ simple-random = {dZ_simple_random:.2f}, dZ complex-random = {dZ_complex_random:.2f})"
            )
        # or favored_complex:
        # if one of the gabor models is favored over random, test which one is favored
        dZ_simple_complex, favored_simple_complex = compare_models(
            log_evidences, models["simple"], models["complex"], min_level=0
        )
        dZ_complex_simple, favored_complex_simple = compare_models(
            log_evidences, models["complex"], models["simple"], min_level=0
        )
        if favored_simple_complex:
            if print_it:
                print(
                    f"Simple model is favored {favored_simple_complex} over complex model with dZ = {dZ_simple_complex:.2f}"
                )
            return "simple"
        elif favored_complex_simple:
            if print_it:
                print(
                    f"Complex model is favored {favored_complex_simple} over simple model with dZ = {dZ_complex_simple:.2f}"
                )
            return "complex"
        else:
            if print_it:
                print(
                    f"No clear winner between simple and complex model (dZ = {dZ_simple_complex:.2f})"
                )
            return "simple"  # if both models are favored over random, but not favored against each other, we can still say that the simple model is favored over random, even if we cannot say that it is favored over the complex model
    else:
        if print_it:
            print("No gabor model is favored over random model.")
        return "random"
