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
##from sift_features import SingleSiftExtractor
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
def protectedDiv1d(left, right):
    if right==0:
        return 0
    else: return left / right

def protectedDiv(left, right):
    with numpy.errstate(divide='ignore',invalid='ignore'):
        x = numpy.divide(left, right)
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = 1
            x[numpy.isnan(x)] = 1
        elif numpy.isinf(x) or numpy.isnan(x):
            x = 1
    return x

def sqrt(left):
    with numpy.errstate(divide='ignore',invalid='ignore'):
        x = numpy.sqrt(left,)
        if isinstance(x, numpy.ndarray):
            x[numpy.isinf(x)] = 1
            x[numpy.isnan(x)] = 1
        elif numpy.isinf(x) or numpy.isnan(x):
            x = 1
    return x

def ifAnd(one,two,three):
    x=0
    if one>0:
        x=two
    elif one<=0:
        x=three
    return x

def relu(left):
    return (abs(left)+left)/2

#----------Filters---------------------------
#convolve----convolved1ed
#convolve(input, weights, output=None, mode='reflect', cval=0.0, origin=0)

#correlate---correlated1d
#correlate(input, weights, output=None, mode='reflect', cval=0.0, origin=0)

#gaussian_filter---gaussian_filter1d
#gaussian_filter(input, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
# An order of 0 corresponds to convolution with a Gaussian kernel. An order of 1, 2, or 3
# corresponds to convolution with the first, second or third derivatives of a Gaussian.
# Higher order derivatives are not implemented
#gaussian filter with sigma=3
#gaussian filter with sigma=3
def gau(left, si):
    return gaussian(left,sigma=si)

def gauD(left, si, or1, or2):
    return ndimage.gaussian_filter(left,sigma=si, order=[or1,or2])

def gau(left, si):
    return ndimage.gaussian_filter(left,sigma=si)

def dog12(left):
    return gau(left, 2)-gau(left, 1)

def dog13(left):
    return gau(left,3)-gau(left, 1)

def gaussian_1(left):
    return ndimage.gaussian_filter(left,sigma=1)
#gaussian filter with sigma=5
def gaussian_2(left):
    return ndimage.gaussian_filter(left,sigma=2)

#gaussian filter with sigma=3
def gaussian_3(left):
    return ndimage.gaussian_filter(left,sigma=3)

#gaussian filter with sigma=1 with the second derivatives
def gaussian_x(left):
    return ndimage.gaussian_filter(left, sigma=1,order=[1,0])
#gaussian filter with sigma=1 with the second derivatives
def gaussian_y(left):
    return ndimage.gaussian_filter(left, sigma=1,order=[0,1])
    
#gaussian filter with sigma=1 with the second derivatives
def gaussian_11(left):
    return ndimage.gaussian_filter(left, sigma=1,order=1)

#gaussian filter with sigma=1 with the second derivatives
def gaussian_12(left):
    return ndimage.gaussian_filter(left, sigma=1,order=2)

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
##    print(numpy.amax(img_filted, axis=0),img_filted.argmax(axis=0).shape)
##    misc.imsave('gobor'+str(round(orientation,2))+name, numpy.amax(img_filted, axis=0))

def gab(left,the,fre):
    fmax=numpy.pi/2
    a=numpy.sqrt(2)
    freq=fmax/(a**fre)
    thea=numpy.pi*the/8
##    print(left)
    filt_real,filt_imag=numpy.asarray(gabor(left,theta=thea,frequency=freq))
##    print(af,af.shape)
    return filt_real

def gabor1(left):
    return gabor_features(left,0)

def gabor2(left):
    return gabor_features(left,numpy.pi/8)

def gabor3(left):
    return gabor_features(left,numpy.pi/4)

def gabor4(left):
    return gabor_features(left,numpy.pi*3/8)

def gabor5(left):
    return gabor_features(left,numpy.pi/2)

def gabor6(left):
    return gabor_features(left,numpy.pi*5/8)

def gabor7(left):
    return gabor_features(left,numpy.pi*3/4)

def gabor8(left):
    return gabor_features(left,numpy.pi*7/8)

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

#percentile_filter(input, percentile, size=None,
#  footprint=None, output=None, mode='reflect', cval=0.0, origin=0)
def percentile(left):
    left=ndimage.percentile_filter(left,percentile=80,size=3)
    return left
#rank_filter(input, rank, size=None,
# footprint=None, output=None, mode='reflect', cval=0.0, origin=0)
#rank=-1, the biggest one; rank=0, the smallest one
def rank(left):
    left=ndimage.rank_filter(left,rank=-1,size=3)
    return left

#prewitt(input, axis=-1, output=None, mode='reflect', cval=0.0)
#[-1,0,1],[-1,0,1],[-1,0,1]
def prewittx(left):
    left=ndimage.prewitt(left,axis=0)
    return left

def prewitty(left):
    left=ndimage.prewitt(left,axis=1)
    return left

def prewittxy(left):
    left=prewitt(left)
    return left

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
def scharrx(left):
    left=scharr_h(left)
    return left

def scharry(left):
    left=scharr_v(left)
    return left

def scharrxy(left):
    left = scharr(left)
    return left
#uniform_filter(input, size=3, output=None,
# mode='reflect', cval=0.0, origin=0)
def uniform(left):
    left=ndimage.uniform_filter(left,3)
    return left

#fourier_ellipsoid(input, size, n=-1, axis=-1, output=None)
def fourier_ellipsoid(left):
    left=ndimage.fourier_ellipsoid(left,size=3)
    return left
# fourier_gaussian(input, sigma, n=-1, axis=-1, output=None)
def fourier_gaussian(left):
    left=ndimage.fourier_gaussian(left,sigma=2,axis=0)
    return left
#fourier_shift(input, shift, n=-1, axis=-1, output=None)
#fourier_uniform(input, size, n=-1, axis=-1, output=None)
def fourier_uniform(left):
    left=ndimage.fourier_uniform(left,size=3,axis=0)
    return left

#morphological_gradient(input, size=None, footprint=None,
#  structure=None, output=None, mode='reflect', cval=0.0, origin=0)
def morphological_gradient(left):
    left=ndimage.morphological_gradient(left,size=3)
    return left
#morphological_laplace(input, size=None, footprint=None,
# structure=None, output=None, mode='reflect', cval=0.0, origin=0
def morphological_laplace(left):
    left=ndimage.morphological_laplace(left,size=3)
    return left

#grey_closing(input, size=None, footprint=None,
# structure=None, output=None, mode='reflect', cval=0.0, origin=0)
def grey_closing(left):
    left=ndimage.grey_closing(left,size=3)
    return left
#grey_dilation(input, size=None, footprint=None,
# structure=None, output=None, mode='reflect', cval=0.0, origin=0)
def grey_dilation(left):
    left=ndimage.grey_dilation(left,size=3)
    return left
#grey_erosion(input, size=None, footprint=None,
# structure=None, output=None, mode='reflect', cval=0.0, origin=0)
def grey_erosion(left):
    left=ndimage.grey_erosion(left,size=3)
    return left
#distance_transform_edt(input, sampling=None, return_distances=True,
#  return_indices=False, distances=None, indices=None)
def distance_transform_edt(left):
    left=ndimage.distance_transform_edt(left)
    return left

#distance_transform_cdt(input, metric='chessboard',
#  return_distances=True, return_indices=False,
# distances=None, indices=None)[source]
def distance_transform_cdt(left):
    left=ndimage.distance_transform_cdt(left)
    return left

#distance_transform_bf(input, metric='euclidean',
#  sampling=None, return_distances=True, return_indices=False,
# distances=None, indices=None
def distance_transform_bf(left):
    left=ndimage.distance_transform_cdt(left)
    return left

def maxPooling(*args):
    left=args[0]
    if len(args) > 1:
        kernelSize = args[1]
    else:
        kernelSize = 3
    M, N=left.shape
    K=kernelSize
    L=kernelSize
    MK=M//K
    NL=N//L
    current = left[:MK * K, :NL * L].reshape(MK, K, NL, L).max(axis=(1, 3))
    #print(current.shape)
    #left=addZerosPad(left,current)
    #print(left.shape)
    return current

def minPooling(*args):
    left=args[0]
    if len(args) > 1:
        kernelSize = args[1]
    else:
        kernelSize = 3
    M, N=left.shape
    #print(left.shape)
    K=kernelSize
    L=kernelSize
    MK=M//K
    NL=N//L
    current = left[:MK * K, :NL * L].reshape(MK, K, NL, L).min(axis=(1, 3))
    #left=addZerosPad(left,current)
    #print(left.shape)
    return current

def stdPooling(*args):
    left=args[0]
    if len(args) > 1:
        kernelSize = args[1]
    else:
        kernelSize = 3
    M, N=left.shape
    #print(left.shape)
    K=kernelSize
    L=kernelSize
    MK=M//K
    NL=N//L
    current = left[:MK * K, :NL * L].reshape(MK, K, NL, L).std(axis=(1, 3))
    #left=addZerosPad(left,current)
    #print(left.shape)
    return current


def addZerosPad(final,current):
    M,N=final.shape
    m1,n1=current.shape
    pUpperSize=int((M-m1)/2)
    pDownSize=int(M-pUpperSize-m1)
    pLeftSize=int((N-n1)/2)
    pRightSize=int(N-pLeftSize-n1)
    PUpper=numpy.zeros((pUpperSize,n1))
    PDown=numpy.zeros((pDownSize,n1))
    current=numpy.concatenate((PUpper,current,PDown),axis=0)
    m2,n2=current.shape
    PLeft=numpy.zeros((m2,pLeftSize))
    PRight=numpy.zeros((m2,pRightSize))
    current=numpy.concatenate((PLeft,current,PRight),axis=1)
    return current

def randomPick(left,index1,index2):
    return left[index1,index2]

#square
def regionS(left,x,y,windowSize):
    #print(left.shape,x,y,windowSize)
    width,height=left.shape
    slice=left[x][y]
    if  (x + windowSize) <=width and (y + windowSize) <=height:
        slice = left[x:(x + windowSize), y:(y + windowSize)]
    elif ((x + windowSize) >width and (y + windowSize) <=height):
        slice = left[x:width, y:(y + windowSize)]
    elif ((x + windowSize) <=width and (y + windowSize) >height):
        slice = left[x:(x + windowSize), y:height]
    elif ((x + windowSize) >width and (y + windowSize) >height):
        slice = left[x:width, y:height]
    #print(slice.shape,x,y,windowSize)
    return slice

#rectangle
def regionR(left, x, y, windowSize1,windowSize2):
    #print(left.shape,x,y,windowSize1,windowSize2)
    width,height=left.shape
    slice=left[x][y]
    if  windowSize1==windowSize2:
        slice=regionS(left, x, y, windowSize1)
    elif (x + windowSize1) <= width and (y + windowSize2) <= height:
        slice = left[x:(x + windowSize1), y:(y + windowSize2)]
    elif ((x + windowSize1) > width and (y + windowSize2) <=height):
        slice = left[x:width, y:(y + windowSize2)]
    elif ((x + windowSize1) <= width and (y + windowSize2) > height):
        slice = left[x:(x + windowSize1), y:height]
    elif ((x + windowSize1) > width and (y + windowSize2) > height):
        slice = left[x:width, y:height]
    #print(slice.shape,x,y,windowSize1,windowSize2)
    return slice

def lbp(image):
    # 'uniform','default','ror','var'
    lbp = local_binary_pattern(image, 8, 1.5, method='nri_uniform')
    lbp=np.divide(lbp,59)
    return lbp

def hist_equal(image):
    equal_image = equalize_hist(image, nbins=256, mask=None)
    return equal_image

def hog_feature(image):
    img, realImage = hog(image, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(3, 3), block_norm='L2-Hys', visualise=True,
                         transform_sqrt=False, feature_vector=True, normalise=None)
    return realImage

def hist_index(image,index):
    n_bins=10
    img=image+1*(image<0)
    hist, ax = numpy.histogram(img,n_bins,[0,1])
    return hist[index]

def hist(image):
    n_bins=10
    img=image+1*(image<0)
    hist, ax = numpy.histogram(img,n_bins,[0,1])
    return hist

def hist_dis(image1,image2):
    hist1=hist(image1)
    hist2=hist(image2)
    distance=numpy.sum(numpy.abs(hist1-hist2))
    return distance

##def sift_f(image1):
##    image=numpy.subtract(image1, 255)
##    image = numpy.asarray(image,dtype=numpy.int8)
##    width,height=image.shape
##    extractor = SingleSiftExtractor(height)
##    feaArrSingle = extractor.process_image(image[:width, :height])
##    print(image.shape,image)
##    print(feaArrSingle)
##    return feaArrSingle

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

def avgP(left, kel1, kel2):
    try:
        current = skimage.measure.block_reduce(left,(kel1,kel2),numpy.mean)
    except ValueError:
        current=left
    #print(current.shape)
    #left=addZerosPad(left,current)
    return current

def ZeroavgP(left, kel1, kel2):
    try:
        current = skimage.measure.block_reduce(left,(kel1,kel2),numpy.mean)
    except ValueError:
        current=left
    #print(current.shape)
    zero_p=addZerosPad(left,current)
    return zero_p

def ZeromaxP(left, kel1, kel2):
    try:
        current = skimage.measure.block_reduce(left,(kel1,kel2),numpy.max)
    except ValueError:
        current=left
    #print(current.shape)
    zero_p=addZerosPad(left,current)
    return zero_p

def maxP2(left):
    kernelSize = 2
    try:
        current = skimage.measure.block_reduce(left,(kernelSize,kernelSize),numpy.max)
    except ValueError:
        current=left
    #print(current.shape)
    #left=addZerosPad(left,current)
    return current

def maxP4(left):
    kernelSize = 4
    try:
        current = skimage.measure.block_reduce(left,(kernelSize,kernelSize),numpy.max)
    except ValueError:
        current=left    
    #print(current.shape)
    #left=addZerosPad(left,current)
    return current

def maxP6(left):
    kernelSize = 6
    try:
        current = skimage.measure.block_reduce(left,(kernelSize,kernelSize),numpy.max)
    except ValueError:
        current=left    
    #print(current.shape)
    #left=addZerosPad(left,current)
    return current

def maxP8(left):
    try:
        current = skimage.measure.block_reduce(left,(8,8),numpy.max)
    except ValueError:
        current=left    
    #print(current.shape)
    #left=addZerosPad(left,current)
    return current

def regionE(img1,img2):
    w1,h1=img1.shape
    w2,h2=img2.shape
    h=max(h1,h2)
    img=numpy.zeros((w1+w2,h))
    img[0:w1,0:h1]=img1
    img[w1:w1+w2,0:h2]=img2
    return img

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

def clusterS(img, nth,windowSize):
          X = numpy.reshape(img, (-1, 1))
          connectivity=grid_to_graph(*img.shape)
          n_clusters=10
          ward=AgglomerativeClustering(n_clusters=n_clusters,
                                       linkage='ward',
                                       connectivity=connectivity)
          ward.fit(X)
##          X[(ward.labels_==0),:]
          c=Counter(ward.labels_)
          value, cound=c.most_common()[nth]
          nth_cluster=X[(ward.labels_==value),:]
          dim=int(numpy.sqrt(len(nth_cluster)))
##          print(np.sqrt(len(nth_cluster)), dim)
          nth_array=numpy.reshape(nth_cluster[0:dim**2], (dim, dim))
##          print(value, cound)
##          print(nth_array.shape)
##          misc.imsave('cluster16.jpg',nth_array)
          return  regionS(nth_array,0,0,windowSize)

def clusterR(img, nth, windowSize1, windowSize2):
          X = numpy.reshape(img, (-1, 1))
          connectivity=grid_to_graph(*img.shape)
          n_clusters=10
          ward=AgglomerativeClustering(n_clusters=n_clusters,
                                       linkage='ward',
                                       connectivity=connectivity)
          ward.fit(X)
##          X[(ward.labels_==0),:]
          c=Counter(ward.labels_)
          value, cound=c.most_common()[nth]
          nth_cluster=X[(ward.labels_==value),:]
          dim=int(numpy.sqrt(len(nth_cluster)))
##          print(np.sqrt(len(nth_cluster)), dim)
          nth_array=numpy.reshape(nth_cluster[0:dim**2], (dim, dim))
##          print(value, cound)
##          print(nth_array.shape)
##          misc.imsave('cluster16.jpg',nth_array)
          return regionR(nth_array,0,0,windowSize1,windowSize2)

def conv_filters(image, filters):
    length = len(filters)
    size = int(sqrt(length))
    filters_resize = numpy.asarray(filters).reshape(size, size)
    img = ndimage.convolve(image, filters_resize)
    return img

def rand_filters(filter_size):
    filters = []
    for i in range(filter_size*filter_size):
        filters.append(numpy.random.randint(-5, 5))
##    print(filters)
    return filters

def addZeros(M, N,current):
    m1,n1=current.shape
    pUpperSize=int((M-m1)/2)
    pDownSize=int(M-pUpperSize-m1)
    pLeftSize=int((N-n1)/2)
    pRightSize=int(N-pLeftSize-n1)
    PUpper=numpy.zeros((pUpperSize,n1))
    PDown=numpy.zeros((pDownSize,n1))
    current=numpy.concatenate((PUpper,current,PDown),axis=0)
    m2,n2=current.shape
    PLeft=numpy.zeros((m2,pLeftSize))
    PRight=numpy.zeros((m2,pRightSize))
    current=numpy.concatenate((PLeft,current,PRight),axis=1)
    return current

def binary_hash(vector):
    s = 0
    for i in range(len(vector)):
        if vector[i]==1:
            s = s+2**i
    return s

def hash_features(*args):
    max_w = 0
    max_h = 0
    for i in args:
        if i.shape[0] > max_w:
            max_w = i.shape[0]
        if i.shape[1] > max_h:
             max_h = i.shape[1]
    images = numpy.zeros((len(args), max_w, max_h))
    for i in range(len(args)):
        images[i,:,:] = addZeros(max_w, max_h,args[i])
    images[images>0]=1
    images[images<=0]=0
    img = numpy.zeros((max_w, max_h))
    for i in range(images.shape[1]):
        for j in range(images.shape[2]):
            vect = images[:, i, j]
            img[i,j] = binary_hash(vect)
##    print(img)
    return img

def hist_block(image, patch_size=8, moving_size=8):
    img=numpy.asarray(image)
    width, height = img.shape
    w = int(width / moving_size)
    h = int(height / moving_size)
    patch = []
    for i in range(0, w):
        for j in range(0, h):
            patch.append([moving_size * i, moving_size * j])
##    print(img.shape, len(patch))
    hist = []
    n_bins = 4
    for i in range(len(patch)):
        small_img = img[patch[i][0]:(patch[i][0] + patch_size), patch[i][1]:(patch[i][1] + patch_size)]
        his,ax=numpy.histogram(small_img,n_bins,[0,n_bins])
##        print(his)
        hist.append(numpy.asarray(his, dtype = float))
    histogram =numpy.vstack(hist)
    histogram = numpy.concatenate((histogram),axis=0)
##    print(histogram.shape, histogram)
    return histogram

def hist_image(image):
    n_bins = 8
    his,ax=numpy.histogram(image,n_bins,[0,n_bins])
    return his
