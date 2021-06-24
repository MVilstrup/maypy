import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline


def prepare_data(data, max_size):
    max_size = len(data) if max_size < 0 else max_size
    max_size = min(len(data), max_size)

    # Convert pandas to numpy
    if str(data.dtype) == 'O':
        data = data.astype(float)
    if 'pandas' in str(type(data)):
        data = data.values

    # Ensure there are no NaN values
    data = data[pd.notna(data)]

    # Make sure its a vector
    data = data.ravel().astype(float)

    # Ensure the correct amount of samples are
    return np.random.choice(data, max_size)


def histogram(X, bins):
    histvals, binedges = np.histogram(X, bins=bins, density=True)
    binedges = (binedges + np.roll(binedges, -1))[:-1] / 2.0
    return binedges, histvals


def smoothline(xs, ys=None, interpol=3, window=1):
    """Smoothing 1D vector.
    Description
    -----------
    Smoothing a 1d vector can be challanging if the number of data is low sampled.
    This smoothing function therefore contains two steps. First interpolation of the
    input line followed by a convolution.
    Parameters
    ----------
    xs : array-like
        Data points for the x-axis.
    ys : array-like
        Data points for the y-axis.
    interpol : int, (default : 3)
        The interpolation factor. The data is interpolation by a factor n before the smoothing step.
    window : int, (default : 1)
        Smoothing window that is used to create the convolution and gradually smoothen the line.
    verbose : int [1-5], default: 3
        Print information to screen. A higher number will print more.
    Returns
    -------
    xnew : array-like
        Data points for the x-axis.
    ynew : array-like
        Data points for the y-axis.
    """

    def smooth(X, window):
        box = np.ones(window) / window
        X_smooth = np.convolve(X, box, mode='same')
        return X_smooth

    if window is not None:
        # Specify number of points to interpolate the data
        # Interpolate
        extpoints = np.linspace(0, len(xs), len(xs) * interpol)
        spl = make_interp_spline(range(0, len(xs)), xs, k=3)
        # Compute x-labels
        xnew = spl(extpoints)
        xnew = xnew[window:-window]

        # First smoothing on the raw input data
        ynew = None
        if ys is not None:
            ys = smooth(ys, window)
            # Interpolate ys line
            spl = make_interp_spline(range(0, len(ys)), ys, k=3)
            ynew = spl(extpoints)
            ynew = ynew[window:-window]
    else:
        xnew, ynew = xs, ys

    return xnew, ynew


def sample_bins(data, bins=50):
    # Calculate Histogram
    data_bins, observations = histogram(data, bins)

    # Smoothing by interpolation
    data_bins, observations = smoothline(data_bins, observations)
    return data_bins, observations
