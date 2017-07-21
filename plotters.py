import numpy as np
import matplotlib.pyplot as plt


def imshowNormalize(data, vmin=None, vmax=None, return_scale=False):
    """ Define a function to scale colour filters

    Useful for plotting 3 channel values that are not normalized between 0 and
    1. The output will be in the range from 0 to 1, so packages like matplotlib
    will display them nicely.

    Parameters
    ----------
    data : ndarray of floats
    vmin : None or float
        Minimum scale value. If set to None, will use the minimum value in data
    vmax : None or float
        Maximum scale value. If set to None, will use the maximum value in data
    return_scale: bool
        If true, will return what the calculated vmin and vmax were

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

    # Ensure numerical stability
    if vmax <= vmin + 0.001:
        vmax = vmin + 0.001

    if return_scale:
        return np.clip((data-vmin)/(vmax-vmin),0,1), vmin, vmax
    else:
        return np.clip((data-vmin)/(vmax-vmin),0,1)


def imshow(data, ax=None):
    """ Imshow with float scaling first

    Calls :py:func:`imshowNormalize` before calling matplotlib's imshow.

    Parameters
    ----------
    data : ndarray of floats
        Data to plot
    ax : matplotlib axis or None
        If None, will get the currently active axis and plot to it. Otherwise,
        can give it the axis object to plot to.
    """
    if ax is None:
        ax = plt.gca()

    ax.imshow(imshowNormalize(data), cmap='gray', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_position([0,0,1,1])


def plot_sidebyside(im1, im2, axes=None):
    """ Plot two images next to each other.

    Calls :py:func:`imshowNormalize` before calling matplotlib's imshow on two
    images.

    Parameters
    ----------
    im1 : ndarray of floats
        Data to plot. Values can be in any range
    im2 : ndarray of floats
        Data to plot. Values can be in any range
    axes : None or array of matplotlib axes of shape (2,)
        The axes to plot im1 and im2 to. If set to None, will create new axes
        and plot to them.
    """
    assert im1.shape == im2.shape

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5),
                                 subplot_kw={'xticks': [], 'yticks': []})
    else:
        assert len(axes) >= 2

    # Create the indices for displaying
    im1_scaled = imshowNormalize(im1)
    im2_scaled = imshowNormalize(im2)
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
    im1 : ndarray of floats
        The initial (fed) image
    im2 : ndarray of floats
        The reconstructed image
    centre : list-like array of floats of shape (2,) or ints of shape (2,)
        If list of is of floats, then they represent the row and col coordinates
        of the centre position in the range 0 to 1 (0 is the left/top and 1 is
        the right/bottom).
        If list of is of ints, then they represent the pixel coordinates for the
        centre position.
    axes : None or array of matplotlib axes of shape (2,)
        The axes to plot im1 and im2 to. If set to None, will create new axes
        and plot to them.
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
    im1_scaled = imshowNormalize(im1[s0,s1,:])
    im2_scaled = imshowNormalize(im2[s0,s1,:])
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
    w : ndarray of floats of shape (x, y, 3, c)
        array of imgs to show
    cols : int
        number of columns to split the filters into
    draw : bool
        True means we will make the calls to plt.imshow, False and we won't,
        only return the big_im array for the caller to handle as they please.
    ax : None or matplotlib axis
        The axis to plot to. If set to None, will create a new axis and plot to
        it (only if draw is True).

    Returns
    -------
    out : tuple of (big_im, vmin, vmax)

        * big_im - Big image, scaled so all the values lie in the range [0,1]
          We do this scaling as matplotlib can't handle well colour images of
          arbitrary ranges.
        * vmin - the minimum value of the big_im (becomes 0 after scaling)
        * vmax - the maximum value of the big_im (becomes 1 after scaling)
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
    big_im, vmin, vmax = imshowNormalize(big_im, return_scale=True)

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


def plot_activations(x, cols=8, draw=True, ax=None, scale_individual=True):
    """Display a 3d tensor as a grid of activations

    Scales the activations and plots them as images.

    Parameters
    ----------
    x : ndarray of floats of shape (x, y, c)
        array of imgs to show
    cols : int
        number of columns to split into
    draw : bool
        True means we will make the calls to plt.imshow, False and we won't,
        only return the big_im array for the caller to handle as they please.
    ax : None or matplotlib axis
        The axis to plot to. If set to None, will create a new axis and plot to
        it (only if draw is True).
    scale_individual : bool
        If true, will scale each of the c activations to be in the range 0 to 1.
        If false, will scale the entire input, x, to be in the range 0 to 1.

    Returns
    -------
    big_img : ndarray of floats of shape (h, w)
        the combined big image array.
    """
    # Calculate the number of rows and columns to display
    nrows = np.int32(np.ceil(x.shape[-1] / cols))

    # New array
    big_im = np.zeros([x.shape[0]*nrows, x.shape[1]*cols])

    # Copy the values across
    for i in range(nrows):
        for j in range(cols):
            if i*cols + j < x.shape[-1]:
                if not scale_individual:
                    big_im[i*x.shape[0]:(i+1)*x.shape[0],
                           j*x.shape[1]:(j+1)*x.shape[1]] = x[:,:,i*cols+j]
                else:
                    big_im[i*x.shape[0]:(i+1)*x.shape[0],
                           j*x.shape[1]:(j+1)*x.shape[1]] = \
                        imshowNormalize(x[:,:,i*cols+j])

    # If we didn't scale already, scale now
    if not scale_individual:
        big_im, vmin, vmax = imshowNormalize(big_im, return_scale=True)

    # Display the image
    if draw:
        if ax is None:
            ax = plt.gca()
            ax.set_position([0, 0, 1, 1])

        ax.imshow(big_im, interpolation='none',cmap='gray')

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
    x : ndarray of floats of shape (x, y, c, 3)
        array of imgs to show
    cols : int
        number of columns to split into
    draw : bool
        True means we will make the calls to plt.imshow, False and we won't,
        only return the big_im array for the caller to handle as they please.
    ax : None or matplotlib axis
        The axis to plot to. If set to None, will create a new axis and plot to
        it (only if draw is True).
    scale_individual : bool
        If true, will scale each of the c activations to be in the range 0 to 1.
        If false, will scale the entire input, x, to be in the range 0 to 1.

    Returns
    -------
    big_img : ndarray of floats of shape (h, w)
        the combined big image array.
    """
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
                        imshowNormalize(x[i*cols+j,:,:,:])
                else:
                    big_im[i*x.shape[1]:(i+1)*x.shape[1],
                           j*x.shape[2]:(j+1)*x.shape[2], :] = \
                        x[i*cols+j,:,:,:]

    # If we didn't scale already, scale now
    if not scale_individual:
        big_im, vmin, vmax = imshowNormalize(big_im, return_scale=True)

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
    x : ndarray of floats of shape (N, height, width, 3)
    centres : ndarray of floats of shape (N, 2) in range [0,1]
        array of centre points. Shape [N, 2]. Float value between 0 and 1
    seize : int
        size in pixels of area to zoom in on

    Returns
    -------
    xzoom : float
        zoomed images
    """
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


def plot_axgrid(h, w, **kwargs):
    """ Create a grid of axes of size h x w

    Creates a figure with tight layout and thin black borders between images.
    Useful for plotting groups of activations

    Parameters
    ----------
    h : int
        number of rows in the grid
    w : int
        number of cols in the grid
    kwargs : (key, val) pairs
        Matplotlib Figure keyword args

    Returns
    -------
    tuple of (fig, axes)
        figure handle and array of matplotlib axes
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
                     'bottom': space, 'top': 1 - space, 'right': 1 - space})
    return fig, axes
