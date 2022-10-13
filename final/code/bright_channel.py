import numpy as np
from skimage import io
from scipy import ndimage
import matplotlib.pyplot as plt
from gf import guided_filter
from pathlib import Path
from argparse import ArgumentParser, Namespace

window_n = 15

def bright_channel( image, k ):
    max_c = np.maximum( np.maximum( image[...,0], image[...,1] ), image[...,2] )
    if k == 1: return max_c
    max_x = ndimage.maximum_filter( max_c, size = k )
    return max_x

def dark_channel( image, k ):
    max_c = np.maximum( np.maximum( image[...,0], image[...,1] ), image[...,2] )
    if k == 1: return max_c
    max_x = ndimage.maximum_filter( max_c, size = k )
    return max_x

# def get_weights( I_max, I_bright ):
#     return (I_bright - I_max) / I_bright
# def L_hat( I_max, I_bright, weight ) :
#     I_bright * (1-weight)+I_max*weight

def enhance(I, L, L_min):
    R = np.zeros( I.shape )
    for i in range(3):
        R[...,i] =  I[...,i] / np.maximum(L, L_min)
    return R

def D_box_filter( image, r ) :
    # image is a 3D array
    h, w, c = image.shape
    out = np.zeros_like( image )
    integral = image.cumsum(0).cumsum(1)

    for i in range( h ) :
        for j in range( w ) :
            left = j - r - 1    if j - r - 1 >= 0   else 0
            right = j + r       if j + r < w        else w - 1
            up = i - r - 1      if i - r - 1 >= 0   else 0
            down = i + r        if i + r < h        else h - 1
            for k in range(3):
                a = integral[up][left][k]
                b = integral[up][right][k]
                c = integral[down][left][k]
                d = integral[down][right][k]
                out[i][j][k] = (d - c - b + a) / ( (right-left) * (down-up) )
    
    return out

def fusion_based(img, I_max, I_bright):
    weight = (I_bright - I_max) / I_bright                  # [0,1]
    L_hat = I_prime_bright = I_bright * ( 1 - weight ) + I_max * weight      # [0, 255]
    L_min = np.ones( (x,y) ) * 50
    return enhance(img, L_hat, L_min), I_prime_bright

def refine(img, R, I_prime_bright):
    D = D_box_filter(R, 1)
    I_min = dark_channel( img, 1 )
    I_dark = dark_channel( img, window_n )
    weight_dark = (I_dark - I_min) / I_dark                                 # [0,1]
    I_prime_dark = I_dark * ( 1 - weight_dark ) + I_min * weight_dark       # [0, 255]
    I_prime_bright  /= 255
    I_prime_dark    /= 255
    s = np.power( np.log( (I_prime_bright+I_prime_dark)/(I_prime_bright-I_prime_dark) ), -0.5 )
    sum_x = np.zeros_like(R)
    for i in range(3):
        sum_x[...,i] = s
    return np.where( D > 0 , R / 255 + (1+sum_x)*D  , R / 255 )


def guided(img, I_bright, resize_r):
    if resize_r : guided_L = np.maximum( np.minimum( guided_filter( img / 255, I_bright, r = window_n*5//resize_r, resize = resize_r, eps = 0.001 ), 1 ), 0 )
    else :  guided_L = np.maximum( np.minimum( guided_filter( img / 255, I_bright, r = window_n*5, resize = resize_r, eps = 0.001 ), 1 ), 0 )
    L_min = np.ones( (x,y) ) * 0.19
    R = enhance(img, guided_L, L_min)
    return np.minimum(R,255)
    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="output image path",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="input image path",
    )
    parser.add_argument(
        "--mode",
        type=int,
        help="0: Fusion_based, 1: Fusion_based_refine, 2: Guided_filter",
        default = 2
    )
    parser.add_argument(
        "--resize",
        type=int,
        default = None
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    resize_r = args.resize
    file_name = args.input
    output_file = args.output
    mode = args.mode
    print(file_name, mode)    
    
    img = io.imread( file_name )
    img = img.astype( np.float32 )
    x,y,c = img.shape
    window_n = 15
    
    I_max = bright_channel( img, 1 )
    I_bright = bright_channel( img, window_n )
    
    if mode == 0:
        result, I_prime_bright = fusion_based(img, I_max, I_bright)
    elif mode == 1:
        I_bright_channel, I_prime_bright = fusion_based(img*255, I_max, I_bright)
        result = refine(img, I_bright_channel, I_prime_bright)
    elif mode == 2:
        result = guided(img, I_bright/255, resize_r) / 255
    else :
        print("Mode should be number in [0, 2]!")
        exit(1)
    
    io.imsave( output_file,  result)
    
