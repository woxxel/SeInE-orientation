import numpy as np
import time
import itertools

from seine.NestedSamplingMethods import (
    run_sampling,
)
from seine import HierarchicalModel, functions as prior_fn
from seine.structures import (
    prior_structure,
)

from .utils import gabor_filter, gabor_response, sine_grating, softplus, ReLU, sigmoid


class HierarchicalBayesInference(HierarchicalModel):

    def set_gratings(self, measure_points, FoV_range, FoV_steps):

        self.measure_points = measure_points

        self.X_FoV, self.Y_FoV = np.meshgrid(
            np.linspace(-FoV_range[0] / 2, FoV_range[0] / 2, FoV_steps),
            np.linspace(-FoV_range[1] / 2, FoV_range[1] / 2, FoV_steps),
        )
        self.FoV_grid = np.dstack((self.X_FoV, self.Y_FoV))

        FoV_steps = self.X_FoV.shape[0]

        self.gratings = np.zeros(
            (
                *[len(mp) for mp in self.measure_points],
                FoV_steps,
                FoV_steps,
            )
        )
        for prod in itertools.product(
            enumerate(self.measure_points[0]),
            enumerate(self.measure_points[1]),
            enumerate(self.measure_points[2]),
        ):
            idx, elems = zip(*prod)
            phi_0, theta, f = elems
            self.gratings[*idx, ...] = sine_grating(
                self.X_FoV,  # [0, ...],
                self.Y_FoV,  # [0, ...],
                theta,
                f,
                phi_0,
                square=True,
            )

    def set_priors(self, priors_init=None, coding="simple"):

        if priors_init is None:

            if hasattr(self, "data"):
                fmap = self.data["observed_counts"] / self.data["T"]
                A0_guess, A_guess = np.percentile(fmap, [50, 90])
                A0_guess = np.maximum(A0_guess, 1.0)
                A_guess = np.maximum(A_guess, 2.0)
            else:
                A0_guess, A_guess = 1.0, 3.0
            self.priors_init = {}
            # print("A guesses from data:", A0_guess, A_guess)

            ## define gabor model priors
            if coding in ["simple", "complex"]:
                self.priors_init["theta"] = prior_structure(
                    prior_fn.bounded_flat,
                    low=-np.pi / 2,
                    high=np.pi / 2,
                    label=r"$\theta$",
                    periodic=True,
                )
                self.priors_init["f"] = prior_structure(
                    prior_fn.halfnorm_ppf,
                    loc=0.0,
                    scale=5.0,
                    label=r"$f$",
                )
                if coding == "simple":
                    ## not required for complex cell model
                    self.priors_init["phi_0"] = prior_structure(
                        prior_fn.bounded_flat,
                        low=-np.pi,
                        high=np.pi,
                        periodic=True,
                        label=r"$\phi_0$",
                    )
                else:
                    self.priors_init["phi_0"] = prior_structure(
                        None,
                        value=0.0,
                        periodic=True,
                        label=r"$\phi_0$",
                    )

                self.priors_init["sigma"] = prior_structure(
                    prior_fn.halfnorm_ppf,
                    loc=0.05,
                    scale=0.1,
                    label=r"$\sigma$",
                )
                self.priors_init["gamma"] = prior_structure(
                    prior_fn.halfnorm_ppf,
                    loc=1.0,
                    scale=1.0,
                    label=r"$\gamma$",
                )
                self.priors_init["theta_gauss"] = prior_structure(
                    prior_fn.bounded_flat,
                    low=-np.pi / 2,
                    high=np.pi / 2,
                    periodic=True,
                    label=r"$\theta_{gauss}$",
                )

            ## define nonlinearity priors - should start with "nl_" to be recognized by model response function
            self.priors_init["nl_baseline"] = prior_structure(
                prior_fn.halfnorm_ppf,
                loc=0.0,
                scale=A0_guess,
                label=r"$A_{baseline}$",
            )
            if coding in ["simple", "complex"]:
                self.priors_init["nl_transition"] = prior_structure(
                    prior_fn.halfnorm_ppf,
                    loc=0.0,
                    scale=1.0,
                    label=r"$A_{transition}$",
                )
                self.priors_init["nl_amplitude"] = prior_structure(
                    prior_fn.halfnorm_ppf,
                    loc=A0_guess * 0.2,
                    scale=A_guess - A0_guess,
                    label=r"$A_{amplitude}$",
                )

            # parameters for loglikelihood
            self.priors_init["logl_alpha"] = prior_structure(
                prior_fn.halfnorm_ppf,
                loc=0.0,
                scale=1.0,
                label=r"$\alpha_{logl}$",
            )

        else:
            self.priors_init = priors_init

        super().set_priors(self.priors_init)

    def get_model_rate_response(self, params, coding="simple"):

        ## transform parameters, if not already provided in proper format
        if not isinstance(params, dict):
            params = self.get_params_from_p(params)

        ## get (normalized) model response
        if coding == "random":
            response = np.zeros(self.dimensions["shape"])
        else:
            response = gabor_response(
                self.X_FoV,
                self.Y_FoV,
                params,
                self.gratings,
                mode=coding,
                significance_threshold=0.01,
                logger=self.log,
            )
        self.timeit("calculating model and response firing rate")

        ## apply nonlinearity to adjust to scale of firing rates and get final model response in firing rate units
        rate_response = self.apply_nonlinearity(response, params)
        return rate_response

    def apply_nonlinearity(self, response, params, nonlinearity="softplus"):

        if nonlinearity == "softplus":
            # print(
            #     f"Applying softplus nonlinearity with parameters: baseline={params.get('nl_baseline', 0.0):.3f}, transition={params.get('nl_transition', 1.0):.3f}, amplitude={params.get('nl_amplitude', 1.0):.3f}"
            # )
            rate_response = softplus(
                response * params.get("nl_amplitude", 1.0),
                alpha=params.get("nl_transition", 1.0),
                delta=params.get("nl_baseline", 0.0),
            )
        elif nonlinearity == "ReLU":
            # rate_response =
            raise NotImplementedError("ReLU nonlinearity not implemented yet")
        elif nonlinearity == "sigmoid":
            raise NotImplementedError("Sigmoid nonlinearity not implemented yet")
        else:
            raise ValueError(f"Unknown nonlinearity: {nonlinearity}")

        self.timeit("applying nonlinearity")
        return rate_response

    def set_logp_func(self, coding="simple"):
        """
        some nice description
        """

        def get_logp(p_in):

            self.timeit()
            """
                build switch between 3 models:
                    - gabor type model
                    - fourier component model (with n components in each direction)
                    - ellipse in rate space
            """

            params = self.get_params_from_p(p_in)
            self.timeit("transforming parameters")

            model_rate_response = self.get_model_rate_response(params, coding)
            if not (model_rate_response.shape == self.data["observed_counts"].shape):
                model_rate_response = model_rate_response[None, ...]

            logp = self.probability_of_spike_observation(
                model_rate_response,
                model="negative_binomial",
                **params,
            )
            self.timeit("calculating log probability of spike observation")

            return logp.sum()

        return get_logp


def run_inference(
    hbm: HierarchicalBayesInference,
    coding="simple",
    n_live=200,
    nP=1,
    dlogz=1.0,
    show_status=True,
):
    hbm.set_priors(coding=coding)
    my_trafo = hbm.set_prior_transform()
    my_logp = hbm.set_logp_func(coding=coding)

    results, sampler = run_sampling(
        my_trafo,
        my_logp,
        hbm.parameter_names_all,
        hbm.periodic,
        show_status=show_status,
        n_live=n_live,
        nP=nP,
        dlogz=dlogz,
    )
    return results


def run_model_comparison(data, dwelltime, measure_points, show_status=True, **kwargs):

    t_start = time.time()
    hbm = HierarchicalBayesInference()

    hbm.set_gratings(
        measure_points,
        [np.deg2rad(140), np.deg2rad(114)],
        FoV_steps=51,
    )

    hbm.prepare_data(
        data,
        dwelltime,
        iter_dims=False,
    )

    results = {}
    for coding in ["random", "simple", "complex"]:
        results[coding] = run_inference(
            hbm, coding=coding, show_status=show_status, **kwargs
        )
    t_end = time.time()
    print(f"Model comparison done after {t_end - t_start:.2f} seconds")
    return results
