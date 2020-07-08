import numpy as np


def gaussian(x_mean, x_std, shape):
    return np.random.normal(x_mean, x_std, shape)


def truncated_gaussian(x_mean, x_std, x_min, shape):
    """
    Samples from a normal distribution, but enforces a minimum value.
    """
    return np.maximum(gaussian(x_mean, x_std, shape), x_min)


def chi2(x_mean, chi2_df, shape):
    """
    Chi-squared distribution centered at a specific mean.
    
    Parameters
    ----------
    x_mean : float
    chi2_df : int
        Degrees of freedom for chi-squared
    shape : list
        Shape of output noise array
        
    Returns
    -------
    dist : ndarray
        Array of chi-squared noise
    """
    return np.random.chisquare(df=chi2_df, size=shape) * x_mean / chi2_df