import numpy as np
import itertools

from event_estimator import get_events


def get_spikes(
    S,
    f=30.05,
):
    spikes = np.zeros_like(S)
    for n, s in enumerate(S):
        spikes[n, :], _, _ = get_events(s, f=f)

    return spikes


def get_unique_stimulus_values(stimulus_data):
    unique_values = {
        "phases": np.unique(stimulus_data["stim_phases"]),
        "angles": np.unique(stimulus_data["stim_angle"]),
        "cycles": np.unique(stimulus_data["stim_cycles"]),
    }
    return unique_values


def calculate_firing_maps(
    stimulus_data,
    spikes,
    f=30.05,
    dt_onset=0.25,  # time to wait after stimulus onset before starting to count spikes
    dt_offset=0.25,
    stimulus_frames=None,
    collapse_repeats=False,
):
    n_cells = spikes.shape[0]

    unique_values = get_unique_stimulus_values(stimulus_data)

    n_stim = (
        len(unique_values["phases"])
        * len(unique_values["angles"])
        * len(unique_values["cycles"])
    )
    if collapse_repeats:
        stimulus_shape = (
            len(unique_values["phases"]),
            len(unique_values["angles"]),
            len(unique_values["cycles"]),
        )
    else:
        n_repeats = stimulus_data["stimulus_clock"].shape[0] // n_stim
        stimulus_shape = (
            n_repeats,
            len(unique_values["phases"]),
            len(unique_values["angles"]),
            len(unique_values["cycles"]),
        )
        print(f"number of unique stimuli: {n_stim}, number of repeats: {n_repeats}")

    event_counts = np.full(stimulus_shape + (n_cells,), 0.0)
    dwelltime = np.full(stimulus_shape, 0.0)

    for prod in itertools.product(
        enumerate(unique_values["phases"]),
        enumerate(unique_values["angles"]),
        enumerate(unique_values["cycles"]),
    ):
        idx, elems = zip(*prod)

        idxes = (
            (stimulus_data["stim_phases"] == elems[0])
            & (stimulus_data["stim_angle"] == elems[1])
            & (stimulus_data["stim_cycles"] == elems[2])
        ).flatten(order="C")
        # print(idxes.shape)
        # idxes = idxes.reshape(1, -1, order="C")[0, :]
        # idxes = idxes.flatten()
        # print(idxes.shape)

        for i, t in enumerate(np.where(idxes)[0]):
            # for t, time in zip(np.where(idxes)[0], times):
            if stimulus_frames is not None:
                start_idx = int(stimulus_frames[t, 0] + dt_onset * f)
                end_idx = int(stimulus_frames[t, 1] + dt_offset * f)
                # end_idx = int(stimulus_frames[t, 1])
            else:
                time = stimulus_data["stimulus_clock"][t, :]
                start_idx = np.argmin(
                    np.abs(stimulus_data["frame_times"] - (time[0] + dt_onset * f))
                )
                end_idx = np.argmin(
                    np.abs(stimulus_data["frame_times"] - (time[1] + dt_offset * f))
                )

            if collapse_repeats:
                event_counts[idx[0], idx[1], idx[2], :] += spikes[
                    :, start_idx:end_idx
                ].sum(axis=1)
                dwelltime[idx[0], idx[1], idx[2]] += (end_idx - start_idx) / f
            else:
                event_counts[i, idx[0], idx[1], idx[2], :] = spikes[
                    :, start_idx:end_idx
                ].sum(axis=1)
                dwelltime[i, idx[0], idx[1], idx[2]] = (end_idx - start_idx) / f

    return event_counts, dwelltime


from matplotlib import pyplot as plt


def plot_spike_maps(
    stimulus_data,
    spikes,
    f=30.05,
    dt_onset=0.25,  # time to wait after stimulus onset before starting to count spikes
    dt_response=0.25,
    stimulus_frames=None,
    neuron=0,
    # collapse_repeats=False,
):
    n_cells = spikes.shape[0]
    stim_duration = np.diff(stimulus_data["stimulus_clock"][:, 0]).mean()

    unique_values = get_unique_stimulus_values(stimulus_data)

    n_stim = (
        len(unique_values["phases"])
        * len(unique_values["angles"])
        * len(unique_values["cycles"])
    )
    # if collapse_repeats:
    #     stimulus_shape = (
    #         len(unique_values["phases"]),
    #         len(unique_values["angles"]),
    #         len(unique_values["cycles"]),
    #     )
    # else:
    n_repeats = stimulus_data["stimulus_clock"].shape[0] // n_stim
    stimulus_shape = (
        n_repeats,
        len(unique_values["phases"]),
        len(unique_values["angles"]),
        len(unique_values["cycles"]),
    )
    print(f"number of unique stimuli: {n_stim}, number of repeats: {n_repeats}")

    event_counts = np.full(stimulus_shape + (n_cells,), 0.0)
    dwelltime = np.full(stimulus_shape, 0.0)

    plt.figure(figsize=(10, 4))
    plt.title("Spike Maps")
    # plt.xlabel("Stimulus Index")
    # plt.ylabel("Neuron Index")

    # for

    for prod in itertools.product(
        enumerate(unique_values["phases"]),
        enumerate(unique_values["angles"]),
        enumerate(unique_values["cycles"]),
    ):
        idx, elems = zip(*prod)

        idxes = (
            (stimulus_data["stim_phases"] == elems[0])
            & (stimulus_data["stim_angle"] == elems[1])
            & (stimulus_data["stim_cycles"] == elems[2])
        ).flatten(order="C")
        # print(idxes.shape)
        # idxes = idxes.reshape(1, -1, order="C")[0, :]
        # idxes = idxes.flatten()
        # print(idxes.shape)

        for i, t in enumerate(np.where(idxes)[0]):
            # print(f"Processing repeat {i}, stimulus indices {idx}, elements {elems}")
            # for t, time in zip(np.where(idxes)[0], times):
            if stimulus_frames is not None:
                start_idx = int(stimulus_frames[t, 0] + dt_onset * f)
                # end_idx = int(stimulus_frames[t, 1] + dt_onset * f)
                end_idx = int(stimulus_frames[t, 1])
            else:
                time = stimulus_data["stimulus_clock"][t, :]
                start_idx = np.argmin(
                    np.abs(stimulus_data["frame_times"] - (time[0] + dt_onset * f))
                )
                end_idx = np.argmin(np.abs(stimulus_data["frame_times"] - time[1]))

            # print(f"Start index: {start_idx}, End index: {end_idx}")

            cts = (spikes[neuron, start_idx:end_idx] > 0).sum().astype(int)
            # print(cts)
            # print(spikes[neuron, start_idx:end_idx])

            offset = idx[1] * stim_duration
            plt.scatter(
                offset + np.where(spikes[neuron, start_idx:end_idx])[0] / f,
                np.full((cts,), idx[2] * n_repeats + i),
                c="b",
                s=1,
            )
            # spikes[:, start_idx:end_idx]
            # if collapse_repeats:
            #     event_counts[idx[0], idx[1], idx[2], :] += spikes[
            #         :, start_idx:end_idx
            #     ].sum(axis=1)
            #     dwelltime[idx[0], idx[1], idx[2]] += (end_idx - start_idx) / f
            # else:
            # event_counts[i, idx[0], idx[1], idx[2], :] = spikes[
            #     :, start_idx:end_idx
            # ].sum(axis=1)
            # dwelltime[i, idx[0], idx[1], idx[2]] = (end_idx - start_idx) / f

    plt.xticks(
        np.linspace(
            0,
            stim_duration * len(unique_values["angles"]),
            len(unique_values["angles"]),
        ),
        labels=unique_values["angles"],
    )
    # return event_counts, dwelltime


def gauss_smooth(X, smooth=None, mode="wrap"):
    if (smooth is None) or not np.any(np.array(smooth) > 0):
        return X
    else:
        V = X.copy()
        V[np.isnan(X)] = 0
        VV = sp.ndimage.gaussian_filter(V, smooth, mode=mode)

        W = 0 * X.copy() + 1
        W[np.isnan(X)] = 0
        WW = sp.ndimage.gaussian_filter(W, smooth, mode=mode)

    return VV / WW
