import numpy
from pylab import *
from scipy import ndimage
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.filters import sobel
from skimage.filters import gabor
import skimage


def relu(left):
    return (abs(left)+left)/2

#
def gauD(left, si, or1, or2):
    img  = []
    for i in range(left.shape[0]):
        img.append(ndimage.gaussian_filter(left[i,:,:],sigma=si, order=[or1,or2]))
    return numpy.asarray(img)

def gau(left, si):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.gaussian_filter(left[i, :, :], sigma=si))
    return numpy.asarray(img)

#gaussian_gradient_magnitude(input, sigma, output=None, mode='reflect', cval=0.0, **kwargs)
def gauGM(left):
    return ndimage.gaussian_gradient_magnitude(left,sigma=1)
#generic_filter
#generic_filter(input, function, size=None, footprint=None, output=None, mode='reflect',
# cval=0.0, origin=0, extra_arguments=(), extra_keywords=None)
#generic_gradient_magnitude
#generic_gradient_magnitude(input, derivative, output=None, mode='reflect',
#cval=0.0, extra_arguments=(), extra_keywords=None)

#gaussian_laplace(input, sigma, output=None, mode='reflect', cval=0.0, **kwargs)
def gaussian_Laplace1(left):
    return ndimage.gaussian_laplace(left,sigma=1)

def gaussian_Laplace2(left):
    return ndimage.gaussian_laplace(left,sigma=2)

#laplace(input, output=None, mode='reflect', cval=0.0)
def laplace(left):
    return ndimage.laplace(left)

###################gabor-filters
def gabor_features(image,orientation):
    image=numpy.asarray(image)
    width,height=image.shape
    fmax=numpy.pi/2
    a=numpy.sqrt(2)
    frequency=[fmax/(a**0), fmax/(a**1),fmax/(a**2),fmax/(a**3),fmax/(a**4)]
##    print(frequency)
    img_filted=numpy.zeros((len(frequency),width,height))
    for j in range(len(frequency)):
##          print(j,frequency[j],orientation)
          filt_real, filt_imag=numpy.asarray(gabor(image,theta=orientation,frequency=frequency[j]))
##          print(filt_real.max(), filt_imag.shape)
          img_filted[j,:,:]=filt_real
##          print(img_filted)
    return numpy.amax(img_filted, axis=0)


def gab(left,the,fre):
    fmax=numpy.pi/2
    a=numpy.sqrt(2)
    freq=fmax/(a**fre)
    thea=numpy.pi*the/8
    img = []
    for i in range(left.shape[0]):
        filt_real,filt_imag=numpy.asarray(gabor(left[i,:,:],theta=thea,frequency=freq))
        img.append(filt_real)
    return numpy.asarray(img)

#maximum_filter(input, size=None,
#  footprint=None, output=None, mode='reflect', cval=0.0, origin=0
def maxf(image):
    img = []
    size = 3
    for i in range(image.shape[0]):
        img.append(ndimage.maximum_filter(image[i,:,:],size))
    return numpy.asarray(img)

#median_filter(input, size=None,
#  footprint=None, output=None, mode='reflect', cval=0.0, origin=0)
def medianf(image):
    img = []
    size = 3
    for i in range(image.shape[0]):
        img.append(ndimage.median_filter(image[i,:,:],size))
    return numpy.asarray(img)

#mean_filter
def meanf(image):
    img = []
    size = 3
    for i in range(image.shape[0]):
        img.append(ndimage.convolve(image[i,:,:], numpy.full((3, 3), 1 / (size * size))))
    return numpy.asarray(img)

#minimum_filter(input, size=None,
# footprint=None, output=None, mode='reflect', cval=0.0, origin=0)
def minf(image):
    img = []
    size = 3
    for i in range(image.shape[0]):
        img.append(ndimage.minimum_filter(image[i,:,:],size))
    return numpy.asarray(img)

def sobelx(left):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.sobel(left[i,:,:], axis=0))
    return numpy.asarray(img)

def sobely(left):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.sobel(left[i, :, :], axis=1))
    return numpy.asarray(img)

def sobelxy(left):
    img = []
    for i in range(left.shape[0]):
        img.append(ndimage.sobel(left[i, :, :]))
    return numpy.asarray(img)

def lbp(image):
    img = []
    for i in range(image.shape[0]):
        # 'uniform','default','ror','var'
        lbp = local_binary_pattern(image[i,:,:], 8, 1.5, method='nri_uniform')
        img.append(numpy.divide(lbp,59))
    return numpy.asarray(img)

def hog_feature(image):
    try:
        img = []
        for i in range(image.shape[0]):
            img1, realImage = hog(image[i, :, :], orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                                transform_sqrt=False, feature_vector=True)
            img.append(realImage)
        data = numpy.asarray(img)
    except: data = image
    return data

def mis_match(img1,img2):
    n, w1,h1=img1.shape
    n, w2,h2=img2.shape
    w=min(w1,w2)
    h=min(h1,h2)
    return img1[:,0:w,0:h], img2[:,0:w,0:h]

def mixadd(img1,img2):
    img11,img22=mis_match(img1,img2)
    return numpy.add(img11,img22)

def mixconadd(img1, w1,img2, w2):
    img11,img22=mis_match(img1,img2)
##    print(w1, w2)
    return numpy.add(img11*w1,img22*w2)

def mixconsub(img1, w1,img2, w2):
    img11,img22=mis_match(img1,img2)
    return numpy.subtract(img11*w1,img22*w2)

def maxP(left, kel1, kel2):
    img = []
    for i in range(left.shape[0]):
        current = skimage.measure.block_reduce(left[i,:,:], (kel1,kel2),numpy.max)
        img.append(current)
    return numpy.asarray(img)

def conta(*args):
    return numpy.asarray(args)

def conVector(img):
    try:
        img_vector=numpy.concatenate((img))
    except:
        img_vector=img
##        print(img_vector)
    return img_vector

def maxP(left, kel1, kel2):
    img = []
    for i in range(left.shape[0]):
        current = skimage.measure.block_reduce(left[i,:,:], (kel1,kel2),numpy.max)
        img.append(current)
    return numpy.asarray(img)

def FeaCon2(img1, img2):
    x_features = []
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        feature_vector = numpy.concatenate((image1, image2), axis=0)
        x_features.append(feature_vector)
    return numpy.asarray(x_features)

def FeaCon3(img1, img2, img3):
    x_features = []
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        image3 = conVector(img3[i, :])
        feature_vector = numpy.concatenate((image1, image2, image3), axis=0)
        x_features.append(feature_vector)
    return numpy.asarray(x_features)


def FeaCon4(img1, img2, img3, img4):
    x_features = []
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :])
        image2 = conVector(img2[i, :])
        image3 = conVector(img3[i, :])
        image4 = conVector(img4[i, :])
        feature_vector = numpy.concatenate((image1, image2, image3, image4), axis=0)
        x_features.append(feature_vector)
    return numpy.asarray(x_features)

def root_conVector1(img1):
    #print(img1.shape,img2.shape)
    x_features = []
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :, :])
        x_features.append(image1)
    return numpy.asarray(x_features)

def root_conVector2(img1, img2):
    #print(img1.shape,img2.shape)
    x_features = []
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :, :])
        image2 = conVector(img2[i, :, :])
        feature_vector = numpy.concatenate((image1, image2), axis=0)
        x_features.append(feature_vector)
    return numpy.asarray(x_features)

def root_conVector3(img1, img2, img3):
    x_features = []
    for i in range(img1.shape[0]):
        image1 = conVector(img1[i, :, :])
        image2 = conVector(img2[i, :, :])
        image3 = conVector(img3[i, :, :])
        feature_vector = numpy.concatenate((image1, image2, image3), axis=0)
        x_features.append(feature_vector)
    return numpy.asarray(x_features)