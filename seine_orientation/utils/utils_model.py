from time import time
from typing import Optional
import logging
import numpy as np


def rotate_2D(X, Y, angle):
    x_rot = X * np.cos(angle) + Y * np.sin(angle)
    y_rot = -X * np.sin(angle) + Y * np.cos(angle)
    return x_rot, y_rot


def gabor_filter(
    X,
    Y,
    # # Parameters (replace with extracted values from your data)
    # amplitude=1.0,  # Amplitude
    theta=np.pi / 4,  # Preferred orientation (radians)
    f=0.5,  # Spatial frequency
    sigma=1.0,  # Gaussian envelope width along x
    gamma=1.0,
    phi_0=0.0,  # Preferred phase (radians)
    theta_gauss=None,
    significance_threshold: Optional[float] = None,
    **kwargs,
):
    # print(
    #     f"gabor_filter called with parameters: theta={theta}, f={f}, sigma={sigma}, gamma={gamma}, phi_0={phi_0}, theta_gauss={theta_gauss}"
    # )

    # Rotate coordinates
    x_prime, y_prime = rotate_2D(X, Y, theta)

    if theta_gauss:
        x_prime_gauss, y_prime_gauss = rotate_2D(x_prime, y_prime, theta_gauss)
    else:
        x_prime_gauss = x_prime
        y_prime_gauss = y_prime

    # Gabor function
    gauss_envelope = np.exp(
        -((x_prime_gauss**2 + gamma**2 * y_prime_gauss**2) / (2 * sigma**2))
    )
    if significance_threshold is not None:
        significant_region = (
            gauss_envelope > gauss_envelope.max() * significance_threshold
        )  # floor to avoid numerical issues
        # print(f"significant fraction: {significant_region.mean():.4f}")
        return (
            gauss_envelope * np.cos(2 * np.pi * f * x_prime + phi_0),
            significant_region,
        )
    else:
        return gauss_envelope * np.cos(2 * np.pi * f * x_prime + phi_0), None


def gabor_response(
    X,
    Y,
    params,
    img,
    mode="simple",
    significance_threshold=None,
    logger: Optional[logging.Logger] = None,
):
    """
    Compute the response of a Gabor filter to an input image.
    Result is normalized to have a maximum of 1, corresponding to the response to the optimal stimulus
    translation into a rate can be done by applying some nonlinearity

    Parameters:
    - X, Y: 2D arrays of coordinates (e.g. from np.meshgrid)
    - params: dict with keys corresponding to gabor_filter parameters (theta, f, sigma, gamma, phi_0, theta_gauss)
    - img: 2D array of the same shape as X/Y representing the input stimulus
    - mode: "simple" for even-symmetric Gabor response, "complex" for energy of even and odd responses
    - significance_threshold: if provided, only consider the region where the Gabor envelope is above this fraction of its maximum (to speed up computation without changing much the response)
    - logger: optional logger for debug output of timing
    """

    def timeit(t_ref, msg=None):
        if logger is None:
            return None
        if msg is not None:  # and (self.time_ref):
            print_msg = f"[GABOR] time for {msg}: {(time() - t_ref) * 10**3:.3f} ms"
            if isinstance(logger, logging.Logger):
                logger.debug(print_msg)
            else:
                print(print_msg)

        return time()

    t_ref = timeit(time())
    if mode == "simple":

        G, significant_region = gabor_filter(
            X, Y, significance_threshold=significance_threshold, **params
        )
        t_ref = timeit(t_ref, f"Gabor filter computed")

        ### normalizing takes too much time and doesnt really change anything
        # G -= G.mean()
        # G /= np.sqrt((G**2).sum() * dx * dy)
        # img -= img.mean()
        # t_ref = timeit(t_ref, f"Gabor and image normalized")

        ## construct proper einsum arguments
        string = ""
        for i in range(len(img.shape) - 2):
            string += chr(ord("a") + i)
        t_ref = timeit(t_ref, f"einsum string defined")

        # optimal_stimulus -= optimal_stimulus.mean()

        if significance_threshold is None:
            einsum_string = f"ij,{string}ij->{string}"
            response = np.einsum(einsum_string, G, img, order="C")
            t_ref = timeit(t_ref, f"Rate response computed")
            ## normalize by optimal response
            optimal_stimulus = sine_grating(X, Y, square=False, **params)
            optimal_response = np.einsum("ij,ij->", G, optimal_stimulus, order="C")
            t_ref = timeit(t_ref, f"Optimal response computed")
        else:
            G = G[significant_region]

            einsum_string = f"i,{string}i->{string}"
            response = np.einsum(
                einsum_string,
                G,
                img[..., significant_region],
                order="C",
            )
            t_ref = timeit(t_ref, f"Rate response computed")
            ## normalize by optimal response
            # optimal_response = np.max(response)
            optimal_stimulus = sine_grating(
                X[significant_region], Y[significant_region], square=False, **params
            )
            # print(f"{G.shape=}, {optimal_stimulus.shape=}")
            optimal_response = np.sum(G * optimal_stimulus)
            t_ref = timeit(t_ref, f"Optimal response computed")
            # print(
            #     f"maximum response: {response.max():.3f}, optimal response: {optimal_response:.3f}"
            # )

        return response / optimal_response
    elif mode == "complex":
        gabor_inputs = {
            "X": X,
            "Y": Y,
            "img": img,
            "mode": "simple",
            "significance_threshold": significance_threshold,
            "logger": logger,
        }
        response_even = gabor_response(params=params, **gabor_inputs)
        response_odd = gabor_response(
            params=params | {"phi_0": params.get("phi_0", 0.0) + np.pi / 2},
            **gabor_inputs,
        )
        response_odd = gabor_response(
            X,
            Y,
            params | {"phi_0": params.get("phi_0", 0.0) + np.pi / 2},
            img,
            mode="simple",
            significance_threshold=significance_threshold,
            logger=logger,
        )
        t_ref = timeit(t_ref, f"sum of even and odd responses computed")
        response = np.sqrt(response_even**2 + response_odd**2)
        t_ref = timeit(t_ref, f"complex rate response computed")
        # print(
        #     f"maximum even response: {response_even.max():.3f}, maximum odd response: {response_odd.max():.3f}, maximum complex response: {response.max():.3f}"
        # )
        return response

    else:
        raise ValueError(f"Unknown mode: {mode}")


def elliptical_pdf(
    X,
    Y,
    x0=0.0,
    y0=0.0,
    sigma_x=1.0,
    sigma_y=1.0,
    angle=0.0,
    amplitude=None,
    return_log=False,
    eps=1e-12,
):
    """
    Evaluate a 2D elliptical Gaussian PDF at coordinates (X, Y).

    Parameters
    - X, Y: array-like of identical shape (can be meshgrid arrays)
    - x0, y0: center position of the ellipse
    - sigma_x, sigma_y: standard deviations along the ellipse principal axes (>0)
    - angle: rotation of the principal axes in radians (counterclockwise)
    - amplitude: if None, uses normalized PDF amplitude 1/(2*pi*sigma_x*sigma_y).
                    If provided, that value is used as a multiplicative prefactor.
    - return_log: if True, return the log-PDF instead of PDF values
    - eps: small floor for sigmas to avoid division by zero

    Returns
    - array of same shape as X/Y with PDF (or log-PDF) values
    """
    sigma_x = max(float(sigma_x), eps)
    sigma_y = max(float(sigma_y), eps)

    # Shift coordinates to center
    dx = X - x0
    dy = Y - y0

    # Precompute sin/cos
    c = np.cos(angle)
    s = np.sin(angle)

    # Components of the inverse covariance matrix Σ^{-1} for rotated Gaussian
    inv_sx2 = 1.0 / (sigma_x * sigma_x)
    inv_sy2 = 1.0 / (sigma_y * sigma_y)

    a = c * c * inv_sx2 + s * s * inv_sy2
    b = s * c * (inv_sx2 - inv_sy2)  # off-diagonal term (will be used as 2*b*dx*dy)
    c_comp = s * s * inv_sx2 + c * c * inv_sy2  # named c_comp to avoid shadowing

    exponent = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c_comp * dy * dy)

    if amplitude is None:
        norm = 1.0 / (2.0 * np.pi * sigma_x * sigma_y)
    else:
        norm = float(amplitude)

    if return_log:
        return np.log(norm) + exponent
    else:
        return norm * np.exp(exponent)


def sine_grating(X, Y, theta, f, phi_0=0.0, square=False, **kwargs):
    # Rotate coordinates
    x_prime = X * np.cos(theta) + Y * np.sin(theta)

    # apply sinusoidal grating
    if square:
        return np.sign(np.cos(2 * np.pi * f * x_prime + phi_0))
    else:
        return np.cos(2 * np.pi * f * x_prime + phi_0)


def softplus(x, alpha=1.0, gamma=0.0, delta=0.0):
    # return alpha * np.log(1 + np.exp((x - gamma) / alpha)) + delta
    return alpha * np.log(1 + np.exp((x - gamma) / alpha)) + delta
    # return alpha * np.log(1 + np.exp((x - gamma) / alpha)) + delta


def ReLU(x, alpha, beta, theta):
    return alpha * np.maximum(beta, x - theta)


def sigmoid(x, alpha=1.0, beta=0.0, theta=0.0):
    return alpha / (1.0 + np.exp(-beta * (x - theta)))


def gompertz_from_specs(M, slope_at_inflect, x_at_small, eps_small):
    """
    Returns a Gompertz y(x) with:
      - Max = M
      - Slope s at inflection
      - y(x_at_small) = eps_small (tiny baseline)
    """
    k = np.e * slope_at_inflect / M
    x0 = x_at_small - (1.0 / k) * np.log(np.log(M / eps_small))

    def y(x):
        return M * np.exp(-np.exp(-k * (x - x0)))

    return y, dict(M=M, k=k, x0=x0)


def gompertz(x, M, k, x0):
    return M * np.exp(-np.exp(-k * (x - x0)))
