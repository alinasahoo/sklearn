import os
os.chdir("C:\Alina\ML IP and CV\ML IP CV Course code\computer_vision")
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# fix random seed for reproducibility
seed_value = 123; np.random.seed(seed_value); #set_random_seed(seed_value) # Keras uses its source of randomness regardless Theano or TensorFlow.In addition, TensorFlow has its own random number generator that must also be seeded by calling the set_random_seed() function immediately after the NumPy random number generator:

exec(open(os.path.abspath('image_common_utils.py')).read())

#%% Basic exploration by reading local image
from skimage.io import imread
img_path = './image_data/elephant.png' # elephant
img = imread(img_path)
img.shape # H x W x C (y,x,c)
img.size # multiplication of HxWxC
type(img)
img.dtype

plt.imshow(img)
plt.show()

# Normalise
x = img/255 # np.max(img), np.min(img)
print(type(x))
print(x.shape) # H x W x C
plt.imshow(x)

# Gray image reading
img = imread(img_path, as_gray=True) 
img.shape # H x W x C
img.size # multiplication of WxHxC
type(img)

plt.imshow(img, cmap=plt.get_cmap('gray')) # cm.gray
plt.show()

#Gray to Black and White
# Pixel range is 0...255, 256/2 = 128
np.min(img), np.max(img)
cutoff = np.max(img)/2
img[img < cutoff] = 0    # Black
img[img >= cutoff] = 1 # White

plt.imshow(img, cmap=plt.get_cmap('gray')) # cm.gray
plt.show()
#CW: You can play with cutoff to get better image

#%% Common function

# Unit test
#img = imread(img_path); show_image(img)
#camera = data.camera(); show_image([img, camera], row_plot = 1)
""" Plot the histogram of an image (gray-scale or RGB) on `ax`. Calculate histogram using `skimage.exposure.histogram`
and plot as filled line. If an image has a 3rd dimension, assume it's RGB and plot each channel separately. """
def plot_histogram(image, ax=None, **kwargs):
    from skimage import exposure

    ax = ax if ax is not None else plt.gca()

    if image.ndim == 2:
        _plot_histogram(ax, image, color='black', **kwargs)
    elif image.ndim == 3:
        # `channel` is the red, green, or blue channel of the image.
        for channel, channel_color in zip(iter_channels(image), 'rgb'):
            _plot_histogram(ax, channel, color=channel_color, **kwargs)

    return

# Helper of above
def _plot_histogram(ax, image, alpha=0.3, **kwargs):
    # Use skimage's histogram function which has nice defaults for integer and float images.
    hist, bin_centers = exposure.histogram(image)
    ax.fill_between(bin_centers, hist, alpha=alpha, **kwargs)
    ax.set_xlabel('intensity')
    ax.set_ylabel('# pixels')
    return
# Helper of above
def iter_channels(color_image):
    """Yield color channels of an image."""
    # Roll array-axis so that we iterate over the color channels of an image.
    for channel in np.rollaxis(color_image, -1):
        yield channel
    return

# Helper of above
def match_axes_height(ax_src, ax_dst):
    """ Match the axes height of two axes objects.
    The height of `ax_dst` is synced to that of `ax_src`.
    """
    # HACK: plot geometry isn't set until the plot is drawn
    plt.draw()
    dst = ax_dst.get_position()
    src = ax_src.get_position()
    ax_dst.set_position([dst.xmin, src.ymin, dst.width, src.height])
    return

""" Plot an image side-by-side with its histogram.
    - Plot the image next to the histogram
    - Plot each RGB channel separately (if input is color)
    - Automatically flatten channels
    - Select reasonable bins based on the image's dtype
    See `plot_histogram` for information on how the histogram is plotted.
    """
def imshow_with_histogram(image, **kwargs):

    width, height = plt.rcParams['figure.figsize']
    fig, (ax_image, ax_hist) = plt.subplots(ncols=2, figsize=(2*width, height))

    kwargs.setdefault('cmap', plt.cm.gray)
    ax_image.imshow(image, **kwargs)
    plot_histogram(image, ax=ax_hist)

    # pretty it up
    ax_image.set_axis_off()
    match_axes_height(ax_image, ax_hist)
    return ax_image, ax_hist
# Unit test
#imshow_with_histogram(img)
#imshow_with_histogram(camera)

# Plot CDF
def plot_cdf(image, ax=None):
    img_cdf, bins = exposure.cumulative_distribution(image)
    ax.plot(bins, img_cdf, 'r')
    ax.set_ylabel("Fraction of pixels below intensity")
    return

#%% Basic exploration by inbuilt images - Explore various Numpy features
#Images manipulated by scikit-image are simply NumPy arrays. Hence, a large fraction of
#operations on images will just consist in using NumPy:
from skimage import data
camera = data.camera()
type(camera)

#Retrieving the geometry of the image and the number of pixels:
camera.shape # (512, 512)
camera.size # 262144

#Retrieving statistical information about gray values:
camera.min(), camera.max(), camera.mean() # (0, 255, 118.3)

#NumPy indexing: NumPy indexing can be used both for looking at pixel values, and to modify
#pixel values:
camera[10, 20] # 153 # Get the value of the pixel on the 10th row and 20th column
camera[3, 10] = 0 # # Set to black the pixel on the 3rd row and 10th column

#Be careful: in NumPy indexing, the first dimension (camera.shape[0]) corresponds to rows, while
#the second (camera.shape[1]) corresponds to columns, with the origin (camera[0, 0]) on the
#top-left corner. This matches  matrix/linear algebra notation, but is in contrast to Cartesian
#(x, y) coordinates. See Coordinate conventions below for more details.

#Beyond individual pixels, it is possible to access / modify values of whole sets of pixels, using
#the different indexing possibilities of NumPy.

#Slicing:
# Set to black the ten first lines
camera[:10] = 0

show_image(camera)

#Masking (indexing with masks of booleans):
mask = camera < 87 # Just random to mask extreme black points

# Set to "white" (255) pixels where mask is True
camera[mask] = 255

show_image(camera)

#Fancy indexing (indexing with sets of indices)
inds_r = np.arange(len(camera))
inds_c = 4 * inds_r % len(camera)
camera[inds_r, inds_c] = 0
show_image(camera)

# SEE THE OUTPUT and then understand the code
#Using masks, especially, is very useful to select a set of pixels on which to perform
#further manipulations. The mask can be any boolean array of same shape as the image (or
#a shape broadcastable to the image shape). This can be useful to define a region of interest,
#such as a disk:
nrows, ncols = camera.shape
#np.ogrid[:5, :5] # Just for example
row, col = np.ogrid[:nrows, :ncols] # which returns an open (i.e. not fleshed out) mesh-grid when
#indexed, so that only one dimension of each returned array is greater than 1.  The dimension and
#number of the output arrays are equal to the number of indexing dimensions.  If the step length
#is not a complex number, then the stop is not inclusive.
row.shape, col.shape
len(row), len(col[0])

# See the output and then understand this logic
cnt_row, cnt_col = nrows / 2, ncols / 2
outer_disk_mask = ((row - cnt_row)**2 + (col - cnt_col)**2 > (nrows / 2)**2)
camera[outer_disk_mask] = 0
show_image(camera)

#Boolean arithmetic can be used to define more complex masks:
lower_half = row > cnt_row
lower_half_disk = np.logical_and(lower_half, outer_disk_mask)

# Take fresh image
camera = data.camera()
camera[lower_half_disk] = 0
show_image(camera)

#Color images
#All of the above is true of color images, too: a color image is a NumPy array, with an
#additional trailing dimension for the channels:
cat = data.chelsea()
type(cat)
cat.shape #(300, 451, 3) This shows that cat is a 300-by-451 pixel image with three channels (red, green, and blue). As before, we can get and set pixel values:
show_image(cat, None)

# See few data
cat[10, 20] #array([151, 129, 115], dtype=uint8) # [red, green, blue]

 # set the pixel at row 51, column 51 to black
cat[50, 50] = 0

# set the pixel at row 51, column 61 to green
cat[50, 60] = [0, 255, 0] # [red, green, blue]
show_image(cat, None) # see on mspaint

#CW: Points may not be visible and hence change for whole column for both black and green cat[50, :]

#We can also use 2D boolean masks for a 2D color image, as we did with the grayscale image above:
cat = data.chelsea()
some_threshold = cat[:, :, 0] > 160
cat[some_threshold] = [0, 255, 0] # RGB
show_image(cat, None) # More greenish

# Convert to black and white
from skimage.color import rgb2gray
from skimage import img_as_float

# First let us convert using both manually and library. See if any difference
cat = data.chelsea()
factor_multiplication = [0.2126, 0.7152, 0.0722] # RGB as per research
cat_manual = img_as_float(cat) @ factor_multiplication # matrix multiplication, @, to convert an RGB image to a grayscale luminance image.
cat_manual.shape
cat_library = rgb2gray(cat)
cat_library.shape

show_image([cat_manual,cat_library], None)

# Error if any
np.mean(np.abs(cat_manual - cat_library)) # 5.0355520774201472e-06

#Summary
#Coordinate conventions:

#In the case of color (or multichannel) images, the last dimension contains the color
#information and is denoted channel or ch.

#Finally, for 3D images, such as videos, magnetic resonance imaging (MRI) scans, or confocal
#microscopy, we refer to the leading dimension as plane, abbreviated as pln or p.

#These conventions are summarized below:
#Dimension name and order conventions in scikit-image
#Image type	coordinates
#2D grayscale	(row, col)
#2D multichannel (eg. RGB)	(row, col, ch)
#3D grayscale	(pln, row, col)
#3D multichannel	(pln, row, col, ch)
#Many functions in scikit-image operate on 3D images directly:

#%%Image Processing by exploring skimage.feature: Count of White dots
#Let’s get started
#Step 1: Import the required library
from skimage.io import imread
from skimage.feature import blob_log # blob_dog, blob_doh

#Step 2 : Import the image
im = imread("./image_data/wint_sky.gif", as_gray=True)
plt.imshow(im, cmap=plt.get_cmap('gray')) # cm.gray
plt.show()

#Step 3 : Find the number of white dots
#Now comes the critical part where our major labor is done by a few commands. These few command go
#out for searching continuous objects in the picture.
# Blobs are found using the Laplacian of Gaussian (LoG) method. For each blob found, the method
#returns its coordinates and the standard deviation of the Gaussian kernel that detected the blob.
blobs_log = blob_log(im, max_sigma=30, num_sigma=10, threshold=.1)
blobs_log.shape

#Blobs_log gives three outputs for each object found. First two are the coordinates and the third one
#is the standard deviation. The radius of each blob/object can be estimated using this column.
#The radius of each blob is approximately sigma * sqrt{2}. Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

# Count the white dots
numrows = len(blobs_log)
print("Number of white dots counted : " ,numrows)

#As we can see that the algorithm has estimated 308 visible white dots. Let’s now look at how accurate are
#these readings.

#Step 4 : Validated whether we captured all the white dots
fig, ax = plt.subplots(1, 1)
plt.imshow(im, cmap=plt.get_cmap('gray'))
for blob in blobs_log:
    y, x, r = blob
    c = plt.Circle((x, y), r+5, color='blue', linewidth=2, fill=False)
    ax.add_patch(c)
plt.show()

del(im, blobs_log)

#Note: It can be done in few other ways. We will learn during this course
#%% Few more exploration
from skimage import img_as_float, img_as_ubyte
from skimage import data

image = data.chelsea()

image_float = img_as_float(image) # Try to use it most of the times
image_ubyte = img_as_ubyte(image)

print("type, min, max, Shape:", image.dtype, image.min(), image.max(), image.shape)
print("type, min, max, Shape:", image_float.dtype, image_float.min(), image_float.max(), image_float.shape)
print("type, min, max, Shape:", image_ubyte.dtype, image_ubyte.min(), image_ubyte.max(), image_ubyte.shape)
print("231/255 =", 231.0/255.0)

#%%Funtime: Some color and circle fun
X, Y = np.ogrid[-5:5:0.1, -5:5:0.1] # Start from -5 with step 0.1 and go till 4.9 (5 - 0.1)
R = np.exp(-(X**2 + Y**2) / 20)

fig, (ax_jet, ax_gray) = plt.subplots(1, 2)
ax_jet.imshow(R, cmap='jet')
ax_gray.imshow(R, cmap='gray')
#%% Color image manipulation
from skimage import data

# Read a sample image
color_image = data.chelsea()
print(color_image.shape)

plt.imshow(color_image)

#Slicing and indexing
#Let's say we want to plot just the red channel of the color image above. We know that the
#red channel is the first channel of the 3rd image-dimension. Since Python is zero-indexed,
#we can write the following:
red_channel = color_image[:, :, 0]  # or color_image[..., 0]

#But when we plot the red channel...
plt.imshow(red_channel) #Obviously that's not red at all.
#CW: What kind of image it is

red_channel.shape # The shape of this array is the same as it would be for any gray-scale image.

#CW: Do for all three colours

#Split this image up into its three components, red, green and blue, and display each separately.
from skimage import io
color_image = io.imread('./image_data/balloon.jpg')
plt.imshow(color_image)

# This code is just a template to get you started.
red_image = color_image[:, :, 0]
green_image = color_image[:, :, 1]
blue_image = color_image[:, :, 2]

show_image([color_image, red_image, green_image, blue_image], row_plot = 1)

#Combining color-slices with row/column-slices
color_patches = color_image.copy()

# Remove green (1) & blue (2) from top-left corner.
color_patches[:100, :100, 1:] = 0
# Remove red (0) & blue (2) from bottom-right corner.
color_patches[-100:, -100:, (0, 2)] = 0
plt.imshow(color_patches)

#%%Histograms: Histograms are a quick way to get a feel for the global statistics of the image
#intensity. For example, they can tell you where to set a threshold or how to adjust the contrast of
#an image. You might be inclined to plot a histogram using matplotlib's hist function:

# Run following to see the error
# plt.hist(color_image) # That didn't work as expected. How would you fix the call above to make it
#work correctly? (Hint: that's a 2-D array, numpy.ravel)
#We will see later. Just FYI as of now

from skimage import exposure

#Histograms of images
camera = data.camera()
imshow_with_histogram(camera)
#An image histogram shows the number of pixels at each intensity value (or range of intensity
#values, if values are binned). Low-intensity values are closer to black, and high-intensity
#values are closer to white. Notice that there's a large peak at an intensity of about 10:
#This peak corresponds with the man's nearly black coat. The peak around the middle of the
#histogram is due to the predominantly gray tone of the image.

#Now let's look at our color image:
cat = data.chelsea()
imshow_with_histogram(cat);
#As you can see, the intensity for each RGB channel is plotted separately. Unlike the previous
#histogram, these histograms almost look like Gaussian distributions that are shifted. This reflects
#the fact that intensity changes are relatively gradual in this picture: There aren't very many
#uniform instensity regions in this image.
#Note: While RGB histograms are pretty, they are often not very intuitive or useful, since a high red
# value is very different when combined with low green/blue values (the result will tend toward red)
# vs high green and blue values (the result will tend toward white).

#Histograms and contrast: Enhancing the contrast of an image allow us to more easily identify features
#in an image, both by eye and by detection algorithms.
#Let's take another look at the gray-scale image from earlier:
camera = data.camera()
imshow_with_histogram(camera)

#Notice the intensity values at the bottom. Since the image has a dtype of uint8, the values go from
# 0 to 255. Though you can see some pixels tail off toward 255, you can clearly see in the histogram,
# and in the image, that we're not using the high-intensity limits very well.

#Based on the histogram values, you might want to take all the pixels values that are more
#than about 180 in the image, and make them pure white (i.e. an intensity of 255). While
#we're at it, values less than about 10 can be set to pure black (i.e. 0). We can do this
#easily using rescale_intensity, from the exposure subpackage.

from skimage import exposure
high_contrast = exposure.rescale_intensity(camera, in_range=(10, 180)) # set to (0, 255)
imshow_with_histogram(high_contrast)

#The contrast is visibly higher in the image, and the histogram is noticeably stretched. The
#sharp peak on the right is due to all the pixels greater than 180 (in the original image)
#that were piled into a single bin (i.e. 255).

#Histogram equalization: In the previous example, the grayscale values (10, 180) were set to
#(0, 255), and everything in between was linearly interpolated. There are other strategies
#for contrast enhancement that try to be a bit more intelligent---notably histogram
#equalization.

#Let's first look at the cumulative distribution function (CDF) of the image intensities.
# Run both line together
ax_image, ax_hist = imshow_with_histogram(camera)
plot_cdf(camera, ax=ax_hist.twinx())

#For each intensity value, the CDF gives the fraction of pixels below that intensity value.
#One measure of contrast is how evenly distributed intensity values are: The dark coat might contrast sharply with the background, but the tight distribution of pixels in the dark coat mean that details in the coat are hidden.
#To enhance contrast, we could spread out intensities that are tightly distributed and combine
#intensities which are used by only a few pixels. This redistribution is exactly what histogram
#equalization does. And the CDF is important because a perfectly uniform distribution gives a CDF
#that's a straight line. We can use equalize_hist from the exposure package to produce an equalized
# image:
equalized = exposure.equalize_hist(camera)
ax_image, ax_hist = imshow_with_histogram(equalized)
plot_cdf(equalized, ax=ax_hist.twinx())

#The tightly distributed dark-pixels in the coat have been spread out, which reveals many details in
#the coat that were missed earlier. As promised, this more even distribution produces a CDF that
#approximates a straight line. Notice that the image intensities switch from 0--255 to 0.0--1.0:
equalized.dtype

#Contrasted-limited, adaptive histogram equalization (CLAHE)
#Unfortunately, histogram equalization tends to give an image whose contrast is artificially
#high. In addition, better enhancement can be achieved locally by looking at smaller patches
#of an image, rather than the whole image. In the image above, the contrast in the coat is
#much improved, but the contrast in the grass is somewhat reduced.
#Contrast-limited adaptive histogram equalization (CLAHE) addresses these issues. The implementation
#details aren't too important, but seeing the result is helpful:
equalized = exposure.equalize_adapthist(camera)
ax_image, ax_hist = imshow_with_histogram(equalized)
plot_cdf(equalized, ax=ax_hist.twinx())

#Compared to plain-old histogram equalization, the high contrast in the coat is maintained,
#but the contrast in the grass is also improved. Furthermore, the contrast doesn't look
#overly-enhanced, as it did with the standard histogram equalization. Again, notice that the
#output data-type is different than our input. This time, we have a uint16 image, which is
#another common format for images:
equalized.dtype

#%%Histograms and thresholding (Separation of foreground from back ground): One of the most
#common uses for image histograms is thresholding.
#Let's return to the original image and its histogram
imshow_with_histogram(camera)

#Here the man and the tripod are fairly close to black, and the rest of the scene is mostly
#gray. But if you wanted to separate the two, how do you decide on a threshold value just
#based on the image? Looking at the histogram, it's pretty clear that a value of about 50
#(guess) will separate the two large portions of this image.
threshold = 50
ax_image, ax_hist = imshow_with_histogram(camera)
ax_image.imshow(camera > threshold)
ax_hist.axvline(threshold, color='red')
# This is a bit of a hack that plots the thresholded image over the original.
# This just allows us to reuse the layout defined in `plot_image_with_histogram`.

#Note that the histogram plotted here is for the image before thresholding.
#This does a pretty good job of separating the man (and tripod) from most of the background.
#Thresholding is the simplest method of image segmentation; i.e. dividing an image into "meaningful"
#regions. More on that later.As you might expect, you don't have to look at a histogram to decide what
#a good threshold value is: There are (many) algorithms that can do it for you. For historical reasons,
#scikit-image puts these functions in the filter module.
#One of the most popular thresholding methods is Otsu's method, which gives a slightly different
#threshold than the one we defined above:

# Rename module so we don't shadow the builtin function
from skimage import filters
threshold = filters.threshold_otsu(camera)
print(threshold)

plt.imshow(camera > threshold)
#Note that the features of the man's face are slightly better resolved in this case.
#You can find a few other thresholding methods in the filter module:

#%%Color spaces: While RGB is fairly easy to understand, using it to detect a specific color
#(other than red, green, or blue) can be a pain. Other color spaces often devote a single
#component to the image intensity (a.k.a. luminance, lightness, or value) and two components
#to represent the color (e.g. hue and saturation in HSL and HSV). You can easily convert to
#a different color representation , or "color space", using functions in the color module.

from skimage import color

# View again
plt.imshow(color_image);

#Here, we'll look at the LAB (a.k.a. CIELAB) color space (L = luminance, a and b define a 2D
# https://en.wikipedia.org/wiki/CIELAB_color_space
#description of the actual color or "hue"):
lab_image = color.rgb2lab(color_image)
color_image.shape, lab_image.shape
color_image[:1,:1,:],lab_image[:1,:1,:]

# Wait - you cannot display above lab image
#The display issues are due to the range of Lab values which are: L (0-100), a (-128 -+127), b (-128 - +127). This really should be documented--our mistake.
#To display a Lab picture, you can rescale the various bands to the desired range (0-1):
lab_image = (lab_image + [0, 128, 128]) / [100, 255, 255]

#Converting to LAB didn't change the shape of the image at all. Let's try to plot it:
plt.imshow(lab_image)

#plot expects an RGB array, and apparently, LAB arrays don't look anything like RGB arrays.
#That said, there is some resemblance to the original.
imshow_all(lab_image[..., 0], lab_image[..., 1], lab_image[..., 2], titles=['L', 'a', 'b'])
show_image([lab_image[..., 0], lab_image[..., 1], lab_image[..., 2]], row_plot = 1)


