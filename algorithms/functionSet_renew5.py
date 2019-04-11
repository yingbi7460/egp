import numpy
import operator
import math
from PIL import Image
from pylab import *
import time
from scipy import ndimage
from skimage.feature import local_binary_pattern
from skimage.exposure import equalize_hist
from skimage.feature import hog
from skimage.filters import median
from skimage.filters import scharr
from skimage.filters import scharr_h
from skimage.filters import scharr_v
from skimage.filters import sobel
from skimage.filters import roberts
from skimage.filters import prewitt
from skimage.filters import gabor
from skimage.filters import gabor_kernel
from skimage.filters import hessian
from skimage.filters import gaussian
import skimage
from collections import Counter
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
#
def gauD(left, si, or1, or2):
    return ndimage.gaussian_filter(left,sigma=si, order=[or1,or2])

def gau(left, si):
    return ndimage.gaussian_filter(left,sigma=si)

def gaussian_1(left):
    return ndimage.gaussian_filter(left,sigma=1)
def gaussian_2(left):
    return ndimage.gaussian_filter(left,sigma=2)

def gaussian_3(left):
    return ndimage.gaussian_filter(left,sigma=3)

def gaussian_x(left):
    return ndimage.gaussian_filter(left, sigma=1,order=[1,0])

def gaussian_y(left):
    return ndimage.gaussian_filter(left, sigma=1,order=[0,1])
    
def gaussian_11(left):
    return ndimage.gaussian_filter(left, sigma=1,order=1)

def gaussian_12(left):
    return ndimage.gaussian_filter(left, sigma=1,order=2)

def gauGM(left):
    return ndimage.gaussian_gradient_magnitude(left,sigma=1)

def gaussian_Laplace1(left):
    return ndimage.gaussian_laplace(left,sigma=1)
def gaussian_Laplace2(left):
    return ndimage.gaussian_laplace(left,sigma=2)

def laplace(left):
    return ndimage.laplace(left)

def gab(left,the,fre):
    fmax=numpy.pi/2
    a=numpy.sqrt(2)
    freq=fmax/(a**fre)
    thea=numpy.pi*the/8
##    print(left)
    filt_real,filt_imag=numpy.asarray(gabor(left,theta=thea,frequency=freq))
##    print(af,af.shape)
    return filt_real

#maximum_filter(input, size=None,
#  footprint=None, output=None, mode='reflect', cval=0.0, origin=0
def maxf(*args):
    """
    :type args: arguments and filter size
    """
    x = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size=3
    x = ndimage.maximum_filter(x,size)
    return x

#median_filter(input, size=None,
#  footprint=None, output=None, mode='reflect', cval=0.0, origin=0)
def medianf(*args):
    """
    :type args: arguments and filter size
    """
    x = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size=3
    x = ndimage.median_filter(x,size)
    return x

#mean_filter
def meanf(*args):
    """
    :type args: arguments and filter size
    """
    x = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size=3
    x = ndimage.convolve(x, numpy.full((3, 3), 1 / (size * size)))
    return x

#minimum_filter(input, size=None,
# footprint=None, output=None, mode='reflect', cval=0.0, origin=0)
def minf(*args):
    """
    :type args: arguments and filter size
    """
    x = args[0]
    if len(args) > 1:
        size = args[1]
    else:
        size=3
    x=ndimage.minimum_filter(x,size)
    return x

#sobel(input, axis=-1, output=None, mode='reflect', cval=0.0)
def sobelx(left):
    left=ndimage.sobel(left,axis=0)
    return left

def sobely(left):
    left=ndimage.sobel(left,axis=1)
    return left

def sobelxy(left):
    left=sobel(left)
    return left
#############################
def fourier_uniform(left):
    left=ndimage.fourier_uniform(left,size=3,axis=0)
    return left


def lbp(image):
    # 'uniform','default','ror','var'
    lbp = local_binary_pattern(image, 8, 1.5, method='nri_uniform')
    lbp=np.divide(lbp,59)
    return lbp
def hog_feature(image):
    img, realImage = hog(image, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(3, 3), block_norm='L2-Hys', visualise=True,
                         transform_sqrt=False, feature_vector=True, normalise=None)
    return realImage

def conta(*args):
    return numpy.asarray(args)

def conta_boolean(*args):
    tf_vector = numpy.zeros((len(args),))
    for i in range(0,len(args)):
        if args[i] < 0:
            tf_vector[i] = 0
        else: tf_vector[i] = 1
    ## print(tf_vector)
    return tf_vector


def mis_match(img1,img2):
    w1,h1=img1.shape
    w2,h2=img2.shape
    w=min(w1,w2)
    h=min(h1,h2)
    return img1[0:w,0:h],img2[0:w,0:h]

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

def mixsubtract(img1,img2):
    img11,img22=mis_match(img1,img2)
    return numpy.subtract(img11,img22)

def mixmultiply(img1,img2):
    img11,img22=mis_match(img1,img2)
    return numpy.add(img11,img22)

def mixprotectedDiv(img1,img2):
    img11,img22=mis_match(img1,img2)
    return protectedDiv(img11,img22)

def conVector(img):
####    print(img.shape)
    try: 
        img_vector=numpy.concatenate((img))
    except ValueError:
        img_vector=img
##        print(img_vector)
    return img_vector

def addvector(vector1, vector2):
    w1=len(vector1)
    w2=len(vector2)
    w=max(len(vector1),len(vector2))
    new_vector1=numpy.zeros((w,))
    new_vector2=numpy.zeros((w,))
    new_vector1[0:len(vector1)]=vector1
    new_vector2[0:len(vector2)]=vector2
    return numpy.add(new_vector1,new_vector2)

def maxP(left, kel1, kel2):
    try:
        current = skimage.measure.block_reduce(left,(kel1,kel2),numpy.max)
    except ValueError:
        current=left
    #print(current.shape)
    #left=addZerosPad(left,current)
    return current

def ZeromaxP(left, kel1, kel2):
    try:
        current = skimage.measure.block_reduce(left,(kel1,kel2),numpy.max)
    except ValueError:
        current=left
    #print(current.shape)
    zero_p=addZerosPad(left,current)
    return zero_p

def root_conVector2(img1, img2):
    image1=conVector(img1)
    image2=conVector(img2)
    feature_vector=numpy.concatenate((image1, image2),axis=0)
##    print(feature_vector.shape)
    return feature_vector

def root_conVector3(img1, img2, img3):
    image1=conVector(img1)
    image2=conVector(img2)
    image3=conVector(img3)
    feature_vector=numpy.concatenate((image1, image2, image3),axis=0)
##    print(feature_vector.shape)
    return feature_vector

def root_conVector4(img1, img2, img3, img4):
    image1=conVector(img1)
    image2=conVector(img2)
    image3=conVector(img3)
    image4=conVector(img4)
    feature_vector=numpy.concatenate((image1, image2, image3, image4),axis=0)
##    print(feature_vector.shape)
    return feature_vector

def root_conVector1(*args):
    feature_vector=numpy.concatenate((args),axis=0)
##    print(feature_vector.shape)
    return feature_vector

def resize(img, w1):
##    print(img.shape, w1)
    img = Image.fromarray(img, 'L')
    img=img.resize((w1,w1),Image.LANCZOS)
    img=numpy.asarray(img)
##    print(img.shape, w1)
    return img
