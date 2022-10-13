import numpy as np
from skimage import io
from scipy import ndimage
from gf import guided_filter
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( 
        "--guided_filter",
        action='store_true'
    )
    parser.add_argument(
        "--input",
        default="img1.jpg",
        type=str
    )
    parser.add_argument(
        "--output",
        default="output_img.jpg",
        type=str
    )
    parser.add_argument(
        "--window_size",
        default=None,
        type=int
    )
    parser.add_argument(
        "--filter_size",
        default=None,
        type=int
    )
    parser.add_argument(
        "--resize",
        default=None,
        type=int
    )
    args = parser.parse_args()
    return args

def dark_channel( image, k ):
    min_c = np.minimum( np.minimum( image[...,0], image[...,1] ), image[...,2] )
    min_x = ndimage.minimum_filter( min_c, size = k )
    return min_x

def get_A( image, dark_channel ) :
    idx = np.unravel_index( np.argsort( dark_channel, axis=None ), dark_channel.shape )
    h, w, c = image.shape
    A = [0, 0, 0]
    for i in range( int( h * w * 0.999 ), int( h * w )  ) :
        y = idx[0][i]
        x = idx[1][i]
        if sum( image[y][x] ) > sum( A ) :
            A = [ image[y][x][j] for j in range( 3 ) ]
    return A
def t_hat( image, k, A ) :
    c = [ image[...,i] / A[i] for i in range( 3 ) ]
    min_c = np.minimum( np.minimum( c[0], c[1] ), c[2] )
    min_x = ndimage.minimum_filter( min_c, size = k )
    return np.ones_like( min_x ) - 0.95 * min_x

def main(args):
    t0 = 0.1
    img = io.imread( args.input )
    img = img.astype( np.float32 )
    if args.window_size is not None :
        r = args.window_size
    else:
        r = min( img.shape[:2] ) / 10
    dc_image = dark_channel( img, r )
    A = get_A( img, dc_image )
    t = t_hat( img, r, A )
    if args.guided_filter :

        if args.filter_size is not None :
            r = args.filter_size
        else :
            r = int( r * 5 )

        if args.resize is not None:
            r = int ( r / args.resize )
        
        t = guided_filter( img / 255, t, r, resize=args.resize )
        t = np.where( t > 1, 1, t )
        t = np.where( t < t0, t0, t )

    for i in range( 3 ): 
        img[...,i] -= A[i]
        img[...,i] /= t
        img[...,i] += A[i]
    img = img / 255
    img = np.where( img > 1, 1, img )
    img = np.where( img < 0, 0, img )
    io.imsave( args.output, img )

if __name__ == "__main__":
    args = parse_args()
    main(args)