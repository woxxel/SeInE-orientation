from __future__ import annotations

import h5py, os
import numpy as np
from pathlib import Path
import multiprocessing as mp
from functools import partial
from scipy.io import loadmat

from pathlib import Path
from typing import Optional

import argparse


def get_folder_path(
    mouse,
    session,
    *,
    type="data",
    dataset="PSD95Mice_Loewel",
    project="tmp",
):
    return Path(project, type, dataset, mouse, session)


def process_session(
    # define the path towards data, assuming hpc structure
    animal: str,
    session: str,
    fname: str,
    dataset: str = "PSD95Mice_Loewel",
    project: str = "orientation_selectivity",
    spikes_key: str = "/estimates/S_dff",
    response_onset: float = 0.2,
    prefix: str = "SeInE",
    suffix: str = "",
    nP: int = 12,
    save_type: str = "hdf5",
    force: bool = False,
    **kwargs,
):
    """
    this assumes some specific structure and naming of data,
    following the HPC project directory conventions.
    """

    from .utils.utils_analysis import (
        calculate_firing_maps,
        get_spikes,
        get_unique_stimulus_values,
    )
    from .BayesModel import (
        run_model_comparison,
    )

    projects_dir = os.environ.get("PROJECT_DIR", "/home/wollex/mnt")
    project_dir = Path(projects_dir) / project

    dir_analysis = get_folder_path(
        animal,
        session,
        type="analysis",
        dataset=dataset,
        project=str(project_dir),
    )
    path_meta = dir_analysis / "CaimanMeta.mat"
    print(path_meta)
    assert path_meta.exists(), f"Metadata file not found: {path_meta}"
    path_detection = dir_analysis / fname
    print(path_detection)
    assert path_detection.exists(), f"Detection file not found: {path_detection}"

    dir_data = get_folder_path(
        animal,
        session,
        type="data",
        dataset=dataset,
        project=str(project_dir),
    )

    ## from here, process data
    ld = loadmat(path_meta, variable_names="CaimanMeta", simplify_cells=True)
    meta_data = ld["CaimanMeta"]
    num_frames = np.cumsum(meta_data["num_frames"])

    with h5py.File(path_detection, "r") as f:
        S = np.array(f[spikes_key][()])

    protocols = {"bino": 0, "cont": 1, "ipsi": 2}

    for protocol, idx in protocols.items():

        fname_out = (
            dir_analysis / f"{prefix}_{protocol}_{Path(fname).stem}{suffix}.{save_type}"
        )

        if num_frames[idx] == 0:
            ## try to obtain from original data
            print(f"Frame number missing ({num_frames}) - obtaining from original TIF data")
            tif_files = sorted(dir_data.glob("*.tif"))

            path_tif_data = tif_files[idx]
            print("TIF path on HPC:", path_tif_data)

            import tifffile

            with tifffile.TiffFile(path_tif_data) as tif:
                s = tif.series[0]
                
                print("shape:", s.shape)
                num_frames[idx] = s.shape[0]
            print("Updated num_frames:", num_frames)


        if Path(fname_out).exists() and not force:
            print(f"Output file for protocol '{protocol}' already exists: {fname_out}")
            continue

        start_idx = num_frames[idx - 1] if idx > 0 else 0
        end_idx = num_frames[idx]

        stimuli = meta_data["Stimulus"][idx]
        S_protocol = S[:, start_idx:end_idx]

        try:
            path_stimulus = dir_data.glob(f"*_{protocol}_*").__next__()
        except StopIteration:
            print(f"Stimulus file for protocol '{protocol}' not found in {dir_data}")
            continue
            # raise FileNotFoundError(f"Stimulus file for protocol '{protocol}' not found in {dir_data}")
        # path_stimulus = dir_data.glob(f"*_{protocol}_*").__next__()
        ld = loadmat(path_stimulus, simplify_cells=True)
        stimulus_data = ld["runInfo"]

        unique_values = get_unique_stimulus_values(stimulus_data)
        measure_points = (
            np.deg2rad(unique_values["phases"]),
            np.deg2rad(unique_values["angles"]),
            1.0 / np.deg2rad(1.0 / unique_values["cycles"]),
        )

        f = meta_data["frame_rate"]
        spikes = get_spikes(S_protocol, f=f)

        ## calculate baseline rates before / after stimulus presentation
        baseline_pre = spikes[:, : int(stimuli[0, 0])].mean(axis=1) * f
        baseline_post = spikes[:, int(stimuli[-1, 1]) :].mean(axis=1) * f

        ## calculate firing maps during stimulus presentation
        event_counts, dwelltime = calculate_firing_maps(
            stimulus_data=stimulus_data,
            spikes=spikes,
            f=f,
            dt_onset=response_onset,
            dt_offset=response_onset / 2,
            stimulus_frames=stimuli[:, :4],
            collapse_repeats=True,
        )

        neurons = range(event_counts.shape[-1])
        idx_process = np.array(neurons)
        n_neurons = len(neurons)

        process_neuron = partial(
            run_model_comparison,
            dwelltime=dwelltime,
            measure_points=measure_points,
            show_status=False,
        )
        # fmaps = (event_counts / dwelltime[..., np.newaxis]).transpose(3, 0, 1, 2)
        fmaps = event_counts.transpose(3, 0, 1, 2)

        batch_sz = 10 * nP
        nBatch = n_neurons // batch_sz

        print(
            f"Processing {n_neurons} neurons in {nBatch+1} batches using {nP} processes..."
        )
        results = []
        with mp.Pool(nP) as pool:
            for i in range(nBatch + 1):
                idx_batch = idx_process[
                    i * batch_sz : min(n_neurons, (i + 1) * batch_sz)
                ]
                outputs = pool.map(
                    process_neuron,
                    fmaps[idx_batch, ...],
                )

                for n, entry in zip(idx_batch, outputs):
                    results.append({})
                    for model in entry.keys():
                        ## store only relevant output
                        results[n][model] = {
                            "fmap": fmaps[n, ...],
                            "baseline": [baseline_pre[n], baseline_post[n]],
                            "dwelltime": dwelltime,  ## is saved n_neuron times, but is not large
                            "evidence": [
                                entry[model].logz[-1],
                                entry[model].logzerr[-1],
                            ],
                            "samples": entry[model].samples,
                            "weights": entry[model].importance_weights(),
                        }

        with h5py.File(fname_out, "w") as f:
            for n, entry in enumerate(results):
                grp_neuron = f.create_group(f"neuron_{n}")
                for model in entry.keys():
                    grp_model = grp_neuron.create_group(model)
                    for key, value in entry[model].items():
                        if isinstance(value, list):
                            grp_model.create_dataset(key, data=np.array(value))
                        else:
                            grp_model.create_dataset(key, data=value)
        print(f"Results from {n_neurons} neurons saved to {fname_out}")


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="seine_orientation",
        description="Orientation selectivity inference using SeInE",
    )
    parser.add_argument(
        "--animal",
        type=str,
        required=True,
        help="Identifier for the animal being processed",
    )
    parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="Identifier for the session being processed",
    )
    parser.add_argument(
        "--fname",
        type=str,
        required=True,
        help="Filename of the imaging data file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="PSD95Mice_Loewel",
        help="Name of the dataset being processed",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="orientation_selectivity",
        help="Name of the project",
    )
    parser.add_argument(
        "--spikes_key",
        type=str,
        default="/estimates/S_dff",
        help="HDF5 key for the spikes data",
    )
    parser.add_argument(
        "--response_onset",
        type=float,
        default=0.2,
        help="Response onset time",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="SeInE",
        help="Prefix of CaImAn result files",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional suffix for different runs of detection",
    )
    parser.add_argument(
        "--nP",
        type=int,
        default=12,
        help="Number of processes for parallel processing",
    )
    parser.add_argument(
        "--save_type",
        type=str,
        default="hdf5",
        help="Specifies result file type (without trailing '.'). Defaults to hdf5",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    process_session(**args.__dict__)

    return 0
