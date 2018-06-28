import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats

__author__ = "Fergal Cotter"
__version__ = "0.0.8"
__version_info__ = tuple([int(d) for d in __version__.split(".")])  # noqa


def imshowNormalize(data, vmin=None, vmax=None, return_scale=False):
    """
    See plotters.normalize
    """
    return normalize(data, vmin, vmax, return_scale)


def normalize(data, vmin=None, vmax=None, return_scale=False,
              make_symmetric=False):
    """ Define a function to scale colour filters

    Useful for plotting 3 channel values that are not normalized between 0 and
    1. The output will be in the range from 0 to 1, so packages like matplotlib
    will display them nicely.

    Parameters
    ----------
    data : ndarray
        The input - numpy array of any shape.
    vmin : None or float
        Minimum scale value. If set to None, will use the minimum value in data
    vmax : None or float
        Maximum scale value. If set to None, will use the maximum value in data
    return_scale: bool
        If true, will return what the calculated vmin and vmax were
    make_symmetric: bool
        If vmin and vmax are None, will ensure that 0.0 is the gray level

    Returns
    -------
    y : scaled_image or tuple
        scaled if return_scale was false - the output scaled data
        (scaled, vmin, vmax) if return_scale was true
    """
    data = data.astype(np.float32)

    if vmin is None:
        vmin = np.min(data)

    if vmax is None:
        vmax = np.max(data)

    if make_symmetric:
        m = max(abs(vmin), vmax)
        vmin = -m
        vmax = m

    # Ensure numerical stability
    if vmax <= vmin + 0.001:
        vmax = vmin + 0.001

    if return_scale:
        return np.clip((data-vmin)/(vmax-vmin),0,1), vmin, vmax
    else:
        return np.clip((data-vmin)/(vmax-vmin),0,1)


def phase_plot(Z, zero_is_black=True):
    """
    Creates an image from the complex array Z.

    Scales Z so that the largest magnitude is 1. The complex argument is used to
    position 3 sinusoids for the R, G, and B channels. When the argument is 0,
    the green channel is largest. When it is 2π/3, the red channel is largest,
    and when it is -2π/3, the blue channel is the largest.

    Parameters
    ----------
    Z : ndarray
        Complex 2d array of numbers
    zero_is_black : bool
        True if we want areas with 0 magnitude to be black. If false, they will
        be grey.

    Returns
    -------
    Y : ndarray
        RGB image of same size as Z.
    """
    im = np.imag(Z)
    re = np.real(Z)

    phase = np.arctan2(im, re)
    amplitude = np.abs(Z)
    amplitude = amplitude/np.max(amplitude)

    # Declare an RGB array
    Y = np.zeros((*Z.shape, 3))
    if zero_is_black:
        Y[:,:,0] = 0.5*amplitude*(np.cos(phase - 2*np.pi/3) + 1)
        Y[:,:,1] = 0.5*amplitude*(np.cos(phase) + 1)
        Y[:,:,2] = 0.5*amplitude*(np.cos(phase + 2*np.pi/3) + 1)
    else:
        Y[:,:,0] = 0.5 + 0.5*amplitude*np.cos(phase - 2*np.pi/3)
        Y[:,:,1] = 0.5 + 0.5*amplitude*np.cos(phase)
        Y[:,:,2] = 0.5 + 0.5*amplitude*np.cos(phase + 2*np.pi/3)

    return Y


def imshow(data, ax=None, **kwargs):
    """ Imshow with float scaling first

    Calls :py:func:`imshowNormalize` before calling matplotlib's imshow.

    Parameters
    ----------
    data : ndarray
        Data to plot.
    ax : None or :py:class:`matplotlib.axes.Axes`.
        If None, will get the currently active axis and plot to it. Otherwise,
        can give it the axis object to plot to.
    kwargs : dict
        Key, value pairs for the axis imshow function to use.

    Notes
    -----
    By default, I use cmap='gray' and interpolation='none' for my imshow kwargs
    as this is what I want to do almost all of the time.
    """
    if ax is None:
        ax = plt.gca()

    defaults = {'cmap': 'gray', 'interpolation': 'none'}
    for key, val in kwargs.items():
        defaults[key] = val

    ax.imshow(normalize(data), **defaults)
    ax.set_xticks([])
    ax.set_yticks([])
    #  ax.set_position([0,0,1,1])


def plot_sidebyside(im1, im2, axes=None):
    """ Plot two images next to each other.

    Calls :py:func:`normalize` before calling matplotlib's imshow on two
    images.

    Parameters
    ----------
    im1 : ndarray
        Data to plot. Values can be in any range
    im2 : ndarray
        Data to plot. Values can be in any range
    axes : None or ndarray(:py:class:`matplotlib.axes.Axes`)
        The axes to plot im1 and im2 to. If set to None, will create new axes
        and plot to them. If an ndarray, should be of shape (2, ).
    """
    assert im1.shape == im2.shape

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5),
                                 subplot_kw={'xticks': [], 'yticks': []})
    else:
        assert len(axes) >= 2

    # Create the indices for displaying
    im1_scaled = normalize(im1)
    im2_scaled = normalize(im2)
    axes[0].imshow(im1_scaled, interpolation='none')
    axes[1].imshow(im2_scaled, interpolation='none')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[0].set_position([0.025, 0, 0.45, 1])
    axes[1].set_position([0.525, 0, 0.45, 1])


def zoom_sidebyside(im1, im2, centre, size, axes=None):
    """ Plot a zoomed in view around a point of two images.

    Will scale the images to be in the range [0,1] before plotting

    Parameters
    ----------
    im1 : ndarray(float)
        The initial (fed) image.
    im2 : ndarray(float)
        The reconstructed image.
    centre : list(int) or list(float)
        If a list is of floats, then interpreted as the row and col coordinates
        of the centre position in the range 0 to 1 (0 is the left/top and 1 is
        the right/bottom).  If list of is of ints, then they represent the pixel
        coordinates for the centre position.
    axes : None or ndarray(:py:class:`matplotlib.axes.Axes`)
        The axes to plot im1 and im2 to. If set to None, will create new axes
        and plot to them. If an ndarray, should be of shape (2, ).
    """
    assert im1.shape == im2.shape

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5),
                                 subplot_kw={'xticks': [], 'yticks': []})
    else:
        assert len(axes) >= 2

    # Create the indices for displaying
    if type(centre[0]) == float or type(centre[1]) == float:
        c = np.array(im1.shape)[0:2] * np.array(centre)
    else:
        c = np.array(centre, np.int)

    assert c[0] > 0 and c[0] < im1.shape[0]
    assert c[1] > 0 and c[1] < im1.shape[1]

    s0 = slice(int(np.maximum(0, c[0]-size/2)),
               int(np.minimum(im1.shape[0], c[0]+size/2)))
    s1 = slice(int(np.maximum(0, c[1]-size/2)),
               int(np.minimum(im1.shape[1], c[1]+size/2)))
    im1_scaled = normalize(im1[s0,s1,:])
    im2_scaled = normalize(im2[s0,s1,:])
    axes[0].imshow(im1_scaled, interpolation='none')
    axes[1].imshow(im2_scaled, interpolation='none')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[0].set_position([0.025, 0, 0.45, 1])
    axes[1].set_position([0.525, 0, 0.45, 1])


def plot_filters_colour(w, cols=8, draw=True, ax=None):
    """ Plot colour filters in a grid

    Display the fiter, w as a grid of n x cols images with no pad
    between images and white grids between each activation.

    Parameters
    ----------
    w : ndarray(float)
        Array of imgs to show. Should be of shape (x, y, 3, c).
    cols : int
        number of columns to split the filters into
    draw : bool
        True means we will make the calls to plt.imshow, False and we won't,
        only return the big_im array for the caller to handle as they please.
    ax : None or :py:class:`matplotlib.axes.Axes`.
        The axis to plot to. If set to None, will create a new axis and plot to
        it (only if draw is True).

    Returns
    -------
    big_im : ndarray
        Big image, scaled so all the values lie in the range [0,1] We do this
        scaling as matplotlib can't handle well colour images of arbitrary
        ranges.
    vmin : float
        The minimum value of the big_im (becomes 0 after scaling)
    vmax : float
        The maximum value of the big_im (becomes 1 after scaling)
    """
    # Calculate the number of rows and columns to display
    nrows = np.int32(np.ceil(w.shape[-1] / cols))

    # New 'big_im' array
    big_im = np.zeros([w.shape[0]*nrows, w.shape[1]*cols, w.shape[2]])

    # Copy the values across
    for i in range(nrows):
        for j in range(cols):
            if i*cols + j < w.shape[-1]:
                big_im[i*w.shape[0]:(i+1)*w.shape[0],
                       j*w.shape[1]:(j+1)*w.shape[1], :] = w[:,:,:,i*cols+j]

    # Calculate the min and max values of the weights to normalize
    big_im, vmin, vmax = normalize(big_im, return_scale=True)

    # Display the image
    if draw:
        if ax is None:
            ax = plt.gca()
            ax.set_position([0, 0, 1, 1])

        ax.imshow(big_im, interpolation='none')

        # Show some gridlines
        # Gridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticks(
            np.arange(-.5, big_im.shape[1]-0.5, w.shape[1]), minor=True)
        ax.set_yticks(
            np.arange(-.5, big_im.shape[0]-0.5, w.shape[0]), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
        ax.tick_params('both', length=0, width=0, which='minor')
        ax.axis('on')

    return big_im, vmin, vmax


def plot_activations(x, cols=8, draw=True, ax=None, scale_individual=True,
                     vmin=None, vmax=None, make_symmetric=False):
    """Display a 3d tensor as a grid of activations

    Scales the activations and plots them as images.

    Parameters
    ----------
    x : ndarray or list(ndarray).
        Array of images to show. Can either be an array of floats of shape
        (x, y, c) or a list of length c of ndarrays of shape (x, y).
    cols : int
        number of columns to split into
    draw : bool
        True means we will make the calls to plt.imshow, False and we won't,
        only return the big_im array for the caller to handle as they please.
    ax : None or :py:class:`matplotlib.axes.Axes`.
        The axis to plot to. If set to None, will create a new axis and plot to
        it (only if draw is True).
    scale_individual : bool
        If true, will scale each of the c activations to be in the range 0 to 1.
        If false, will scale the entire input, x, to be in the range 0 to 1.
    vmin : float or None
        Value to set as the negative limit (black). If None, will calculate
        from data. Ignored if scale_individual is True.
    vmax : float or None
        Value to set as the positive limit (white). If None, will calculate
        from data. Ignored if scale_individual is True.

    Returns
    -------
    big_im : ndarray(floats)
        The combined big image array.
    """
    if type(x) is list:
        x = np.stack(x, axis=-1)

    # Calculate the number of rows and columns to display
    nrows = np.int32(np.ceil(x.shape[-1] / cols))

    # New array
    big_im = np.zeros([x.shape[0]*nrows, x.shape[1]*cols])

    # Copy the values across
    for i in range(nrows):
        for j in range(cols):
            if i*cols + j < x.shape[-1]:
                if scale_individual:
                    big_im[i*x.shape[0]:(i+1)*x.shape[0],
                           j*x.shape[1]:(j+1)*x.shape[1]] = \
                        normalize(x[:,:,i*cols+j],
                                  make_symmetric=make_symmetric)
                else:
                    big_im[i*x.shape[0]:(i+1)*x.shape[0],
                           j*x.shape[1]:(j+1)*x.shape[1]] = x[:,:,i*cols+j]

    # If we didn't scale already, scale now
    if not scale_individual:
        big_im, vmin, vmax = normalize(big_im, vmin, vmax,
                                       return_scale=True)

    # Display the image
    if draw:
        if ax is None:
            ax = plt.gca()
            ax.set_position([0, 0, 1, 1])

        ax.imshow(big_im, interpolation='none',cmap='gray',vmin=0, vmax=1)

        # Show some gridlines
        # Gridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticks(
            np.arange(-.5, big_im.shape[1]-0.5, x.shape[1]), minor=True)
        ax.set_yticks(
            np.arange(-.5, big_im.shape[0]-0.5, x.shape[0]), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='r', linestyle='-', linewidth=0.5)
        ax.tick_params('both', length=0, width=0, which='minor')
        ax.axis('on')

    return big_im


def plot_batch_colour(x, cols=8, draw=True, ax=None, scale_individual=True):
    """ Plot a batch of colour images/patches.

    Display x as a grid of n x cols images with no pad between images and red
    grids between each activation.


    Parameters
    ----------
    x : ndarray or list(ndarray)
        Array of images to show. If input is an ndarray, should be of shape
        (batch, x, y, 3). Can also be a list of length `batch`, with each entry
        being an ndarray of shape (x, y, 3)
    cols : int
        number of columns to split into
    draw : bool
        True means we will make the calls to plt.imshow, False and we won't,
        only return the big_im array for the caller to handle as they please.
    ax : None or :py:class:`matplotlib.axes.Axes`
        The axis to plot to. If set to None, will create a new axis and plot to
        it (only if draw is True).
    scale_individual : bool
        If true, will scale each of the c activations to be in the range 0 to 1.
        If false, will scale the entire input, x, to be in the range 0 to 1.

    Returns
    -------
    big_im : ndarray
        The combined big image array.
    """
    if type(x) is list:
        x = np.stack(x, axis=0)

    # If x is grayscale (of shape [n, x, y]), stack it to be rgb
    if len(x.shape) == 3:
        x = np.stack([x, x, x], axis=-1)

    # Calculate the number of rows and columns to display
    nrows = np.int32(np.ceil(x.shape[0] / cols))

    # New 'big_im' array
    big_im = np.zeros([x.shape[1]*nrows, x.shape[2]*cols, 3])

    # Copy the values across
    for i in range(nrows):
        for j in range(cols):
            if i*cols + j < x.shape[0]:
                if scale_individual:
                    big_im[i*x.shape[1]:(i+1)*x.shape[1],
                           j*x.shape[2]:(j+1)*x.shape[2], :] = \
                        normalize(x[i*cols+j,:,:,:])
                else:
                    big_im[i*x.shape[1]:(i+1)*x.shape[1],
                           j*x.shape[2]:(j+1)*x.shape[2], :] = \
                        x[i*cols+j,:,:,:]

    # If we didn't scale already, scale now
    if not scale_individual:
        big_im, vmin, vmax = normalize(big_im, return_scale=True)

    # Display the image
    if draw:
        if ax is None:
            ax = plt.gca()
            ax.set_position([0, 0, 1, 1])
        ax.imshow(big_im, interpolation='none')
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticks(
            np.arange(-.5, big_im.shape[1]-0.5, x.shape[2]), minor=True)
        ax.set_yticks(
            np.arange(-.5, big_im.shape[0]-0.5, x.shape[1]), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='r', linestyle='-', linewidth=1)
        ax.tick_params('both', length=0, width=0, which='minor')
        ax.axis('on')

    return big_im


def zoom_batch_colour(x, centres, size=10):
    """Zoom in around a pixel for an array of images.

    Parameters
    ----------
    x : ndarray or list(ndarray)
        Input. Can be an numpy array floats of shape (N, height, width, 3) or
        list of length N of numpy arrays of floats, each of shape (height,
        width, 3).
    centres : ndarray
        Array of centre points. Shape must be (N, 2) and the values should be
        floats between 0 and 1.
    size : int
        size in pixels of area to zoom in on

    Returns
    -------
    xzoom : float
        One big image made up of the patches of zoomed images from the input.
    """
    if type(x) is list:
        x = np.stack(x, axis=0)

    # Check if the centres was a single tuple/list
    centres = np.array(centres)
    if centres.shape == (2,):
        centres = np.tile(centres, [x.shape[0], 1])

    # Check that the centres matches the size of x
    assert x.shape[0] == centres.shape[0]
    c = np.zeros_like(centres, dtype=np.int32)
    c[:,0] = (centres[:,0] * x.shape[1]).astype(np.int32)
    c[:,1] = (centres[:,1] * x.shape[2]).astype(np.int32)

    # Make sure size is even
    size = int(size + (size % 2))
    # Halfsize
    halfsize = int(size/2)

    # Zoom in on x
    xzoom = np.zeros((x.shape[0], size, size, 3),dtype=np.float32)
    for i in range(x.shape[0]):
        # Handle cases where the centre could be close to the image border
        # by moving the centre slightly inwards.
        if c[i,0] < size/2:
            s0 = slice(0, size)
        elif c[i,0] > x.shape[1] - halfsize:
            s0 = slice(x.shape[1]-size, x.shape[1])
        else:
            s0 = slice(c[i,0] - halfsize, c[i,0] + halfsize)

        if c[i,1] < size/2:
            s1 = slice(0, size)
        elif c[i,1] > x.shape[2] - halfsize:
            s1 = slice(x.shape[2]-size, x.shape[2])
        else:
            s1 = slice(c[i,1]-halfsize, c[i,1]+halfsize)

        # Now zoom in on the image
        try:
            xzoom[i,:,:,:] = x[i, s0, s1, :]
        except:
            print('image {}, size {} & {} & {}'.format(i, s0, s1, c[i]))
            raise ValueError('Incompatible centres')

    return xzoom


def plot_axgrid(h, w, top=1, **kwargs):
    """ Create a grid of axes of size h x w

    Creates a figure with tight layout and thin black borders between images.
    Useful for plotting groups of activations

    Parameters
    ----------
    h : int
        Number of rows in the grid.
    w : int
        Number of cols in the grid.
    top : float
        Extent of the subfigures at the top. Useful to give space for titles and
        what not.
    kwargs : (key, val) pairs
        :py:class:`matplotlib.figure.Figure` keyword args.

    Returns
    -------
    fig : :py:class:`matplotlib.figure.Figure`
    axes : :py:class:`matplotlib.axes.Axes`
    """
    space = 0.02
    default_args = {
        'facecolor': 'k'
    }
    for key, val in kwargs.items():
        default_args[key] = val

    fig, axes = plt.subplots(
        h, w, **default_args,
        subplot_kw={'xticks': [], 'yticks': []},
        gridspec_kw={'hspace': space, 'wspace': space, 'left': space,
                     'bottom': space, 'top': top - space, 'right': 1 - space})
    return fig, axes


def plot_dtcwt(yl, yh, fig=None, f=np.abs, top=1, fmt='chw', imshow_kwargs={}):
    """ Plot the dtcwt coefficients of an image on a single figure

    Parameters
    ----------
    yl : ndarray
        Lowpass output
    yh : list(ndarray)
        Complex bandpass outputs. Can be (h, w, 6) or (6, h, w)
    fig : None or matplotlib.Figure
        Figure to plot to (will create one if None)
    f : callable
        Function to apply to highpasses to convert their outputs to real numbers
    top : float
        Top of the figure in relative coordinates. I.e. between 0 and 1. Set to
        1 by default so plots take up full height, but can reduce this if you
        want to add a title.
    fmt : str
        Either 'chw' or 'hwc' depending on the format of yh
    """
    J = len(yh)

    space = 0.02
    if fig is None:
        fig = plt.figure(facecolor='k')

    if 'cmap' not in imshow_kwargs.keys():
        imshow_kwargs['cmap'] = 'viridis'

    widths = [0.5, 0.1, 1, 1, 1, 1, 1, 1]
    gs = gridspec.GridSpec(J+1, 8, hspace=space*2, wspace=space,
                           left=space+0.05, bottom=space, top=top-space,
                           right=1-space, width_ratios=widths)

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    gradient = gradient.T[::-1]

    # Preprocess the data
    yh_disp = [f(scale) for scale in yh]

    for j in range(J):
        vmin = yh_disp[j].min()
        vmax = yh_disp[j].max()
        hist = fig.add_subplot(gs[j,0], xticks=[])
        cmap = fig.add_subplot(gs[j,1], xticks=[], yticks=[])

        # Plot the histogram of data
        if f == np.abs:
            x = np.geomspace(1, vmax+1, 50) - 1
        else:
            x = np.linspace(vmin, vmax, 50)
        # Fit a kernel to it
        try:
            density = stats.gaussian_kde(yh_disp[j].ravel())
        except np.linalg.LinAlgError:
            def delta(x):
                if x == 0:
                    return 1
                else:
                    return 0
            density = np.vectorize(delta)
        y = density(x)
        # Plot it vertically and then make it right-aligned
        hist.set_xlim(y.max(), 0)
        # Fill in the space between the curve and the axis
        hist.fill_between(y, 0, x)

        # Plot the colourmap
        cmap.imshow(gradient, cmap=imshow_kwargs['cmap'], aspect='auto')

        for i in range(6):
            ax = fig.add_subplot(gs[j,i+2], xticks=[], yticks=[])
            if fmt.lower() == 'chw':
                ax.imshow(yh_disp[j][i], vmin=vmin, vmax=vmax, **imshow_kwargs)
            else:
                ax.imshow(yh_disp[j][:,:,i], vmin=vmin, vmax=vmax,
                          **imshow_kwargs)

    # Plot the lowpass
    vmin = yl.min()
    vmax = yl.max()
    hist = fig.add_subplot(gs[J,0], xticks=[])
    cmap = fig.add_subplot(gs[J,1], xticks=[], yticks=[])

    x = np.linspace(vmin, vmax, 50)
    # Fit a kernel to it
    density = stats.gaussian_kde(yl.ravel())
    y = density(x)
    # Plot it vertically and then make it right-aligned
    hist.set_xlim(y.max(), 0)
    # Fill in the space between the curve and the axis
    hist.fill_between(y, 0, x)

    # Plot the colourmap
    cmap.imshow(gradient, cmap=imshow_kwargs['cmap'], aspect='auto')
    ax = fig.add_subplot(gs[J,2], xticks=[], yticks=[])
    ax.imshow(yl, **imshow_kwargs)
