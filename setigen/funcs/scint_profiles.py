import numpy as np
from scipy.stats import norm

from setigen.funcs import func_utils


def autocorrelation(x, length=20):
    # Returns up to length index shifts for autocorrelations
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0, 1]
                         for i in range(1, length)])


def get_rho(ts, tscint, p):
    '''
    Get autocorrelations with time array ts and scintillation
    timescale tscint.
    '''
    # Calculate sigma from width
    sigma = tscint / (2 * np.sqrt(2 * np.log(2)))
    y = func_utils.gaussian(ts, (ts[0] + ts[-1]) / 2, sigma)
    rho = autocorrelation(y, length=p+1)[1:]
    return rho


def psi(r):
    '''Return covariance matrix for initial multivariate normal distribution'''
    # r is the array of guesses to get close to desired autocorrelations
    p = len(r)
    covariance = np.ones((p, p))
    for i in range(0, p - 1):
        for j in range(0, p - i - 1):
            covariance[i + j + 1, j] = covariance[j, i + j + 1] = r[i]
    return covariance


def build_Z(r, T):
    '''Build full baseline Z array'''
    # T is final length of array Z, should be greater than p
    # r is the array of guesses to get close to desired autocorrelations
    # Returns full Z array
    p = len(r)
    assert T >= p

    Z = np.zeros(T)
    covariance = psi(r)

    Z[:p] = np.random.multivariate_normal(np.zeros(p), covariance)
    alpha = np.dot(r, np.linalg.inv(covariance))
    assert np.all(np.abs(np.roots([1]+list(-alpha))) <= 1.)

    variance = 1 - np.dot(alpha, r)
    assert variance >= 0

    for i in range(p, T):
        epsilon = np.random.normal(0, np.sqrt(variance))
        Z[i] = np.dot(alpha, Z[i-p:i][::-1]) + epsilon
    return Z


def inv_exp_cdf(x, rate=1):
    '''Inverse exponential distribution CDF'''
    return -np.log(1. - x) / rate


def get_Y(Z):
    '''
    Get final values specific to an overall exponential distribution,
    normalized to mean of 1.
    '''
    Y = inv_exp_cdf(norm.cdf(Z))
    return Y / np.mean(Y)


def scint_t_profile(Y, level=1):
    def t_profile(t):
        if type(t) is np.ndarray:
            assert len(Y) == t.shape[0]
            return np.repeat(Y.reshape((t.shape[0], 1)) * level, t.shape[1],
                             axis=1)
        elif type(t) is list:
            return Y[:len(t)]
        else:
            return 0
    return t_profile
