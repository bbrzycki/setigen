import numpy as np
from numpy.testing import assert_array_equal
import pytest

import setigen as stg


def test_quantize_real_zero_std_returns_target_mean():
    values = np.array([5.0, 5.0, 5.0])

    quantized = stg.voltage.quantization.quantize_real(values,
                                                       target_mean=3,
                                                       target_std=2,
                                                       num_bits=8,
                                                       data_mean=5.0,
                                                       data_std=0.0)

    assert_array_equal(quantized, np.array([3, 3, 3]))


def test_real_quantizer_stats_cache_period():
    quantizer = stg.voltage.RealQuantizer(target_fwhm=8,
                                          num_bits=8,
                                          stats_calc_period=2,
                                          stats_calc_num_samples=8)
    values = np.linspace(-1, 1, 8)

    quantizer.quantize(values)
    first_cache = tuple(quantizer.stats_cache)
    assert quantizer.stats_calc_indices == 1

    quantizer.quantize(values + 10)
    assert tuple(quantizer.stats_cache) == first_cache
    assert quantizer.stats_calc_indices == 0

    quantizer.quantize(values + 10)
    assert quantizer.stats_cache[0] != first_cache[0]


def test_complex_quantizer_scalar_and_pair_custom_stds():
    quantizer = stg.voltage.ComplexQuantizer(target_fwhm=8,
                                             num_bits=8,
                                             stats_calc_period=1,
                                             stats_calc_num_samples=8)
    values = np.array([1 + 2j, 3 + 4j, -1 - 2j])

    scalar_quantized = quantizer.quantize(values, custom_stds=2.0)
    pair_quantized = quantizer.quantize(values, custom_stds=[2.0, 3.0])

    assert scalar_quantized.shape == values.shape
    assert pair_quantized.shape == values.shape
    assert np.iscomplexobj(scalar_quantized)
    assert np.iscomplexobj(pair_quantized)


def test_complex_quantizer_invalid_custom_stds_length():
    quantizer = stg.voltage.ComplexQuantizer(target_fwhm=8,
                                             num_bits=8)
    values = np.array([1 + 1j, 2 + 2j])

    with pytest.raises(ValueError, match="length-2 sequence"):
        quantizer.quantize(values, custom_stds=[1.0, 2.0, 3.0])


def test_complex_quantizer_reset_cache():
    quantizer = stg.voltage.ComplexQuantizer(target_fwhm=8,
                                             num_bits=8,
                                             stats_calc_period=2,
                                             stats_calc_num_samples=8)
    values = np.array([1 + 1j, 2 + 2j, 3 + 3j])

    quantizer.quantize(values)
    assert quantizer.quantizer_r.stats_cache[0] is not None
    assert quantizer.quantizer_i.stats_cache[0] is not None

    quantizer._reset_cache()
    assert quantizer.stats_cache_r == [None, None]
    assert quantizer.stats_cache_i == [None, None]
    assert quantizer.quantizer_r.stats_cache == [None, None]
    assert quantizer.quantizer_i.stats_cache == [None, None]
