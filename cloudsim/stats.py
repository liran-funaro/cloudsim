"""
Author: Liran Funaro <liran.funaro@gmail.com>

Copyright (C) 2006-2018 Liran Funaro

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import scipy.stats
from scipy.stats import *
import numpy as np
import matplotlib.pyplot as plt
import itertools
import warnings


def trunc_norm(min_val, max_val, mean, std, size=None, increasing=False):
    a, b = (min_val - mean) / std, (max_val - mean) / std
    y = truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
    if size and increasing:
        y.sort()
    return y


def trunc_norm_cdf(min_val, max_val, mean, std, size):
    a, b = (min_val - mean) / std, (max_val - mean) / std
    x = np.linspace(min_val, max_val, size)
    y = truncnorm.cdf(x, a, b, loc=mean, scale=std)
    return x, y


def trunc_norm_ppf(min_val, max_val, mean, std, size):
    a, b = (min_val - mean) / std, (max_val - mean) / std
    x = np.linspace(0, 1, size)
    return truncnorm.ppf(x, a, b, loc=mean, scale=std)


def trunc_norm_obj(min_val, max_val, mean, std):
    a, b = (min_val - mean) / std, (max_val - mean) / std
    return truncnorm(a, b, loc=mean, scale=std)


def get_beta_params_mean(mean, density=1, max_val=1):
    eps = np.finfo(np.float32).eps
    mean = np.divide(mean, max_val)
    mean = np.clip(mean, eps, 1 - eps)
    beta_a = density
    beta_b = density
    if mean < 0.5:
        beta_a = np.maximum(density * mean / (1 - mean), eps)
    else:
        beta_b = np.maximum(density * (1 - mean) / mean, eps)
    return beta_a, beta_b, 0, max_val


def get_beta_params_mean_vec(mean, density=1, max_val=1):
    # E = a / (a+b) => E*(a+b) = a => E*a + E*b = a => E*b = a - E*a => E*b = a*(1-E)
    # => a = b * E/(1-E)
    # => b = a * (1-E)/E
    eps = np.finfo(np.float32).eps
    size = len(mean)
    mean = np.divide(mean, max_val)
    mean = np.clip(mean, eps, 1 - eps)
    u = mean < 0.5
    beta_a = np.repeat(density, size).astype(float)
    beta_b = np.repeat(density, size).astype(float)
    beta_a[u] = np.maximum(density * mean[u] / (1 - mean[u]), eps)
    beta_b[~u] = np.maximum(density * (1 - mean[~u]) / mean[~u], eps)
    return beta_a, beta_b, 0, max_val


def get_beta_trunc_mean(mean, density, max_val, trunc):
    a, b, loc, scale = get_beta_params_mean(mean, density, max_val=max_val)
    p = scipy.stats.beta.sf(trunc, a, b, loc=loc, scale=scale)
    return scipy.stats.beta.expect(args=(a, b), loc=loc, scale=scale, ub=trunc) + p*trunc


def search_rising_func(func, v, xl, xh, *args, max_iter=32, full_output=False, **kwargs):
    yl, yh = (func(x, *args, **kwargs) for x in (xl, xh))
    if v > yh:
        return xh, yh, 0
    elif v < yl:
        return xl, yl, 0

    xm = 0
    ym = 0
    i = 0

    for i in range(np.maximum(max_iter, 1)):
        weight = (yh - v) / (yh - yl)
        xm = weight * xl + (1 - weight) * xh
        ym = func(xm, *args, **kwargs)
        if np.isclose(ym, v):
            break
        elif ym > v:
            xh, yh = xm, ym
        else:
            xl, yl = xm, ym

    if full_output:
        return xm, ym, i + 1
    else:
        return xm


def multi_outer(vecs):
    """ Computes outer product of multiple arrays """
    ret = np.array(vecs[0])
    for v in vecs[1:]:
        a = ret.reshape((*ret.shape, 1))
        # b = np.array(v).reshape(1,len(v))
        b = np.array([v])
        ret = np.matmul(a, b)
    return ret


def greater_or_close(a, b, rtol=1e-05, atol=1e-08):
    """ numpy isclose(): absolute(a - b) <= (atol + rtol * absolute(b)) """
    return (b - a) < (atol + rtol * np.abs(b))


#####################################################################
# Statistical Information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CDF: Cumulative distribution function
#   F(x) = Pr(X<x)
#   F(b) - F(a) = Pr(a<X<b)
# SF: Survival function
#   S(x) = 1-F(x)
#   S(x) = Pr(X>x)
#   S(a) - S(b) = Pr(a<X<b)
#   S(x) + F(x) = 1
# PDF: Probability Density Function
#   f(x) = F'(x) = -S'(x)
#####################################################################


def cdf_to_pdf(cdf, dx=1, axis=-1):
    return np.gradient(cdf, dx, axis=axis)


def pdf_to_cdf(pdf, dx=1, axis=-1):
    nd = pdf.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)

    sub_pdf = np.divide(pdf[slice1] + pdf[slice2], 2.0)
    sub_pdf = np.insert(sub_pdf, 0, 0, axis=axis)
    return np.cumsum(sub_pdf, axis=axis) * dx


def vec_mean(vec, axis=-1):
    nd = vec.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    return np.divide(vec[slice1] + vec[slice2], 2.0)


def cdf_to_sf(cdf):
    return np.subtract(1, cdf)


def sf_to_cdf(cdf):
    return np.subtract(1, cdf)


def integrate_pdf(pdf, dx=1, axis=-1):
    return np.trapz(pdf, dx=dx, axis=axis)


def integrate_pdf_2d(pdf, dx=1):
    ret = pdf[1:-1, 1:-1].sum()
    ret += pdf[1:-1, 0].sum() / 2.
    ret += pdf[1:-1, -1].sum() / 2.
    ret += pdf[0, 1:-1].sum() / 2.
    ret += pdf[-1, 1:-1].sum() / 2.
    ret += (pdf[0, 0] + pdf[0, 1] + pdf[1, 0] + pdf[-1, -1]) / 4.
    ret *= dx
    return ret


def cdf_minimum(cdf1, cdf2):
    return cdf1 + cdf2 - cdf1*cdf2


def cdf_maximum(cdf1, cdf2):
    return cdf1 * cdf2


def cdf_shift(cdf, shift, min_val=0, max_val=1):
    cdf_length = len(cdf)
    dx = (max_val - min_val) / (cdf_length-1)
    return np.insert(cdf, 0, [0] * int(shift / dx))[:cdf_length]


def pdf_minimum(pdf1, pdf2, cdf1=None, cdf2=None, dx=1):
    if cdf1 is None:
        cdf1 = pdf_to_cdf(pdf1, dx)
    if cdf2 is None:
        cdf2 = pdf_to_cdf(pdf2, dx)
    return pdf1 + pdf2 - pdf1 * cdf2 - pdf2 * cdf1


def expected_via_cdf(cdf, min_val=0, max_val=1, axis=-1):
    cdf_length = cdf.shape[axis]
    if cdf_length <= 1:
        return min_val
    dx = (max_val - min_val) / (cdf_length - 1)
    return np.trapz(cdf_to_sf(cdf), dx=dx, axis=axis) + min_val


def percentile_via_cdf(cdf, percentile, min_val=0, max_val=1):
    cdf_x = np.linspace(min_val, max_val, len(cdf))
    return np.interp(percentile, cdf, cdf_x)


def find_cdf_limit_that_yields_mean(cdf, min_val=0, max_val=1, required_mean=0.5, axis=-1):
    ndim = cdf.ndim
    if axis < 0:
        axis = ndim + axis

    cdf_length = cdf.shape[axis]
    if cdf_length <= 1:
        return min_val
    dx = (max_val - min_val) / (cdf_length - 1)
    sf = cdf_to_sf(cdf)
    mean_of_cdf = np.full_like(cdf, min_val, dtype=float)
    ii = (np.s_[:], ) * axis
    kk = (np.s_[:], ) * (ndim - axis - 1)
    mean_of_cdf[ii + np.s_[2:, ] + kk] += np.cumsum(
        (sf[ii + np.s_[1:-1, ] + kk] + sf[ii + np.s_[:-2, ] + kk]) * dx / 2.0, axis=axis)
    mean_of_cdf[ii + np.s_[1:, ] + kk] += sf[ii + np.s_[:-1, ] + kk] * dx / 2.0
    i = np.abs(mean_of_cdf - required_mean).argmin(axis=axis)
    return dx * i


def cdf_from_sample(sample, cdf_size=1024, x_limits=None, cdf_x=None):
    sample = np.sort(sample)
    if cdf_x is None:
        if x_limits is None:
            x_limits = sample[0], sample[-1]
        l, h = x_limits
        if l is None:
            l = sample[0]
        if h is None:
            h = sample[-1]
        if l > h:
            l, h = h, l
        cdf_x = np.linspace(l, h, cdf_size)

    cdf_x_vals = cdf_x - np.finfo('float32').eps
    sample_vals = np.linspace(0, 1, len(sample))

    # Numpy interp is 4 times faster than scipy.interpolate.interp1d
    # even without copy and assume sorted.
    cdf_y = np.interp(cdf_x_vals, sample, sample_vals, left=0, right=1)
    return cdf_x, cdf_y


def ppf_from_cdf(sample_x, cdf_x, cdf_y):
    return np.interp(sample_x, cdf_y, cdf_x)


def sample_from_cdf(cdf_x, cdf_y, sample_count):
    uniform_sample = np.random.uniform(0, 1, sample_count)
    return ppf_from_cdf(uniform_sample, cdf_x, cdf_y)


def calc_hist_bin_edges(centers):
    """
    Finds the histogram bin edges given its centers
    """
    if len(centers) > 1:
        d = np.diff(centers) / 2
    else:
        d = np.array([5])
    edges = [centers[0] - d[0], *(centers[:-1] + d), centers[-1] + d[-1]]
    return np.array(edges)


def calc_hist_bin_centers(edges):
    """
    Finds the histogram bin centers given its edges
    """
    if len(edges) < 2:
        return edges
    return (edges[1:] + edges[:-1]) / 2


def intify_hist(hist_x, hist_y, expected_count, expected_sum, max_sum, atol=1e-08, max_iter=1024):
    """
    Find the closest histogram integer values to the input histogram
    """
    buckets_count = len(hist_x)

    hist_y_floor = np.floor(hist_y).astype(int)
    count_diff = expected_count - hist_y_floor.sum()
    if np.isclose(count_diff, 0):
        return hist_y_floor

    diff = hist_y - hist_y_floor

    buckets_p = diff / count_diff

    count_opt = [int(np.floor(count_diff)), int(np.ceil(count_diff))]
    dec = count_diff - np.floor(count_diff)
    count_p = [1 - dec, dec]

    floor_sum = (hist_x * hist_y_floor).sum()

    best_result = hist_y_floor
    best_diff = np.abs((hist_x * best_result).sum() - expected_sum)

    for _ in range(max_iter):
        count = np.random.choice(count_opt, 1, p=count_p)
        c = np.random.choice(buckets_count, count, p=buckets_p, replace=False)

        r = (hist_x[c].sum() + floor_sum)
        cur_diff = np.abs(r - expected_sum)
        if r <= max_sum and cur_diff < best_diff:
            best_diff = cur_diff
            best_result = np.copy(hist_y_floor)
            for i in c:
                best_result[i] += 1
            if cur_diff < atol:
                break

    return best_result


def get_scipy_dist(args):
    args_count = len(args)
    if args_count < 2:
        raise ValueError("args must have at least two arguments (name and params)")
    dist_name, dist_param = args[:2]

    if type(dist_param) not in [tuple, list]:
        dist_param = (dist_param,)

    dist = getattr(scipy.stats, dist_name)
    return dist, dist_param


def scipy_dist_ppf(ppf_x, args):
    dist, dist_param = get_scipy_dist(args)
    if len(args) > 3:
        ppf_x_loc, ppf_x_scale = args[3]
        ppf_x = (ppf_x * ppf_x_scale) + ppf_x_loc

    ret = dist.ppf(ppf_x, *dist_param)
    if len(args) > 2:
        clip = args[2]
        ret = np.clip(ret, clip[0], clip[1])
    return ret


def scipy_dist_cdf(cdf_x, args):
    dist, dist_param = get_scipy_dist(args)
    return dist.cdf(cdf_x, *dist_param)


def scipy_dist_correlated(input_data, args):
    dist_name = args[0]
    dist = getattr(scipy.stats, dist_name)

    func = eval(args[1])
    s_param = args[2]
    all_param = []
    for p in s_param:
        if type(p) in [tuple, list]:
            v = func(input_data, *p)
        else:
            v = np.full(len(input_data), p)
        all_param.append(v)

    ret = np.array([dist.rvs(*p[:-2], loc=p[-2], scale=p[-1]) for p in zip(*all_param)])

    if len(args) > 3:
        clip = args[3]
        ret = np.clip(ret, clip[0], clip[1])
    return ret


def graphic_fit(util, title, dist_names=None, guess=(), reduce_sample=1024, limits=None):
    """
    'alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford', 'burr', 'cauchy', 'chi', 'chi2',
    'cosine', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponweib', 'exponpow', 'f', 'fatiguelife',
    'fisk', 'foldcauchy', 'foldnorm', 'frechet_r', 'frechet_l', 'genlogistic', 'genpareto', 'genexpon',
    'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'gilbrat', 'gompertz', 'gumbel_r',
    'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'hypsecant', 'invgamma', 'invgauss',
    'invweibull', 'johnsonsb', 'johnsonsu', 'ksone', 'kstwobign', 'laplace', 'logistic', 'loggamma', 'loglaplace',
    'lognorm', 'lomax', 'maxwell', 'mielke', 'nakagami', 'ncx2', 'ncf', 'nct', 'norm', 'pareto', 'pearson3',
    'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'reciprocal', 'rayleigh', 'rice', 'recipinvgauss',
    'semicircular', 't', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'wald',
    'weibull_min', 'weibull_max', 'wrapcauchy'
    """
    warnings.filterwarnings('ignore')

    cdf_x, cdf_y = cdf_from_sample(util, 1024, x_limits=limits)
    plt.plot(cdf_x, cdf_y, label="ECDF")

    if type(reduce_sample) == int and reduce_sample > 2:
        util_x = np.linspace(0, 1, len(util))
        sub_x = np.linspace(0, 1, reduce_sample)
        sub_util = np.interp(sub_x, util_x, np.sort(util))
    else:
        sub_util = util

    if dist_names is None:
        dist_names = ['beta', 'powerlognorm', 'weibull_min']

    best_param = None
    best_err = float('inf')
    best_ind = -1

    for i, (dist_name, param) in enumerate(itertools.chain([(d, None) for d in dist_names], guess)):
        dist = getattr(scipy.stats, dist_name)
        if param is None:
            param = list(dist.fit(sub_util))
        else:
            dist_name = "Guess: %s" % dist_name
            param = list(param)

        cdf_fitted = dist.cdf(cdf_x, *param[:-2], loc=param[-2], scale=param[-1])
        y_diff = cdf_y - cdf_fitted
        err = np.sqrt((y_diff * y_diff).sum())

        if err < best_err:
            best_err = err
            best_param = np.array(param)
            best_ind = i

        print(dist_name, "error: %.2f" % err, "params:", param)
        plt.plot(cdf_x, cdf_fitted, label=dist_name)

    plt.title(title)
    plt.legend()
    print("Best ind:", best_ind)

    return best_param


def concave_convex_plot(x, y, force='convex'):
    i = 0
    eps = 1e-8
    if force == 'convex':
        diff_func = lambda d:  (d[1:] + eps > d[:-1])
    else:
        diff_func = lambda d: (d[1:] < d[:-1] + eps)

    while True:
        i += 1
        diff = np.diff(x) / np.diff(y)
        b = diff_func(diff)
        if np.sum(b) == 0:
            break
        a = np.full_like(x, True, dtype=bool)
        a[1:-1] = ~b
        x = x[a]
        y = y[a]

    return x, y