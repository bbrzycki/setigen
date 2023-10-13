from __future__ import absolute_import, division, print_function

from setigen._version import __version__

from setigen.funcs import (
    constant_path, squared_path, sine_path, simple_rfi_path,
    constant_t_profile, sine_t_profile, periodic_gaussian_t_profile,
    box_f_profile, gaussian_f_profile, multiple_gaussian_f_profile,
    lorentzian_f_profile, voigt_f_profile, sinc2_f_profile,
    constant_bp_profile
)
from setigen import voltage

from setigen.distributions import fwhm, gaussian, truncated_gaussian, chi2
from setigen.waterfall_utils import (
    max_freq, min_freq, get_data, get_fs, get_ts
)
from setigen.sample_from_obs import (
    sample_gaussian_params, get_parameter_distributions, get_mean_distribution
)
from setigen.split_utils import (
    split_waterfall_generator, split_fil, split_array
)
from setigen.stats import (
    exclude_and_flatten, get_mean, get_std, get_min, compute_frame_stats
)
from setigen.unit_utils import cast_value, get_value
from setigen.frame_utils import db, array, integrate, get_slice
from setigen.normalize import sigma_clip_norm, sliding_norm, max_norm
from setigen.dedrift import dedrift
from setigen.plots import plot_frame, plot_cadence

from setigen.frame import Frame, params_from_backend
from setigen.cadence import Cadence, OrderedCadence