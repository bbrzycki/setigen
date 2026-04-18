from __future__ import annotations

import importlib

import numpy as np


def _get_array_module(backend):
    if backend == "numpy":
        return np
    if backend == "cupy":
        cupy = importlib.import_module("cupy")
        return cupy
    if backend == "auto":
        try:
            cupy = importlib.import_module("cupy")
        except ImportError:
            return np
        return cupy
    raise ValueError(f"Unsupported reduction backend '{backend}'.")


def _fftshifted_spectra(voltages, *, fftlength, xp):
    trimmed = voltages[:(voltages.shape[0] // fftlength) * fftlength]
    if trimmed.shape[0] == 0:
        return None

    spectra_by_pol = []
    for pol in range(trimmed.shape[2]):
        samples = xp.asarray(trimmed[:, :, pol]).T
        samples = samples.reshape((samples.shape[0], samples.shape[1] // fftlength, fftlength))
        fft_vals = xp.fft.fft(samples, fftlength, axis=2) / fftlength**0.5
        fft_vals = xp.fft.fftshift(fft_vals, axes=2)
        spectra_by_pol.append(fft_vals.transpose(1, 0, 2))
    return spectra_by_pol


def _integrate_products(products, *, integration_factor, xp):
    trimmed = products[:(products.shape[0] // integration_factor) * integration_factor]
    if trimmed.shape[0] == 0:
        return None
    trimmed = trimmed.reshape((trimmed.shape[0] // integration_factor, integration_factor, *trimmed.shape[1:]))
    return trimmed.sum(axis=1)


def _flatten_channels(products):
    return products.reshape(products.shape[0], products.shape[1] * products.shape[2])


def _channelize_block(voltages,
                      *,
                      fftlength,
                      integration_factor,
                      pol_mode,
                      backend):
    xp = _get_array_module(backend)
    spectra = _fftshifted_spectra(voltages, fftlength=fftlength, xp=xp)
    if spectra is None:
        return None

    xx = xp.abs(spectra[0]) ** 2
    yy = xp.abs(spectra[1]) ** 2 if len(spectra) > 1 else None

    if pol_mode == 1:
        total = xx if yy is None else xx + yy
        integrated = _integrate_products(total, integration_factor=integration_factor, xp=xp)
        if integrated is None:
            return None
        result = _flatten_channels(integrated)[:, np.newaxis, :]
    else:
        if yy is None:
            raise ValueError("Full-pol and full-Stokes reduction require a dual-polarization RAW input.")
        xy = spectra[0] * xp.conj(spectra[1])
        re_xy = xp.real(xy)
        im_xy = xp.imag(xy)

        if pol_mode == 4:
            products = xp.stack((xx, yy, re_xy, im_xy), axis=1)
        elif pol_mode == -4:
            products = xp.stack((xx + yy,
                                 xx - yy,
                                 2 * re_xy,
                                 -2 * im_xy), axis=1)
        else:
            raise ValueError(f"Unsupported polarization mode '{pol_mode}'.")

        integrated = _integrate_products(products, integration_factor=integration_factor, xp=xp)
        if integrated is None:
            return None
        result = integrated.reshape(integrated.shape[0], integrated.shape[1], integrated.shape[2] * integrated.shape[3])

    if xp is not np:
        result = xp.asnumpy(result)
    return np.asarray(result, dtype=np.float32)
