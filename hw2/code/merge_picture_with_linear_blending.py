from argparse import ArgumentParser, Namespace
from pathlib import Path
from skimage import io
from skimage import color, filters, transform
import numpy as np
import matplotlib.pyplot as plt

def img_read_np_cp( pic_num, path ):
    img_np = []
    img_cp = []
    for i in range( pic_num ):
        file_name = path / f"img{'{0:02d}'.format(i)}.JPG"
        img = io.imread( file_name )
        print( "read : ", file_name )
        img_np.append( transform.resize( img, scale_tuple( img.shape, 0.5 ) ) )
    return img_np, img_cp

def scale_tuple( t, n ) :
    return ( t[0] * n, t[1] * n, t[2] )

def calculate_offset( off, size ):
    h, w = size
    print( "pic size ( h, w ) : ", h, w )
    off = [[int(i[0] / 2), int(i[1] / 2)] for i in off]
    for i in range( 1, pic_nums ) :
        off[i][0] = off[i][0] + off[i-1][0]
        off[i][1] = off[i][1] + off[i-1][1]
    off_max_h = max( [ off[i][1] for i in range( pic_nums ) ] )
    off_min_h = min( [ off[i][1] for i in range( pic_nums ) ] )
    off_max_w = off[pic_nums - 1][0]
    output_size_h = h + off_max_h - off_min_h
    output_size_w = w + off_max_w
    print( "output_size ( h, w ): ", output_size_h, output_size_w )
    output_shape = np.array( [ output_size_h, output_size_w ] )    
    off_matrix = []
    for i in range( pic_nums ):
        off[i][1] -= off_min_h
        off_mtx = np.array( [[1, 0, -off[i][0]], [0, 1, -off[i][1]], [0, 0, 1]] )
        off_matrix.append( off_mtx )
    print( "off_matrix complete" )
    return output_shape, off_matrix, off

def add_alpha(img, background = -1):
    alpha = ( img[:,:,0] != background )
    return np.dstack( (img, alpha) )

def linear_blending(img, width, offset, i):
    if i > 0:
        # linear blending foreward
        A = offset[i-1][0] + width[i-1]
        B = offset[i][0]
        margin_len = A - B
        # B is 0%, A is 100%
        for p in range(B, A):
            img[:,p:p+1,:] = img[:,p:p+1,:] * ((p - B) / margin_len)

    if i < len(offset)-1:
        # linear blending afterward
        A = offset[i][0] + width[i]
        B = offset[i+1][0]
        margin_len = A - B
        # B is 100%, A is 0%
        for p in range(B, A):
            img[:,p:p+1,:] = img[:,p:p+1,:] * ((A - p) / margin_len)


def Scissors(offsetA, offsetB, width):
    return min(width, offsetB - offsetA + 100)

def main( args ) :
    global pic_nums
    pic_nums = args.photo_nums
    img_np, img_cp = img_read_np_cp( pic_nums,  args.photo_dir )
    f = args.offset
    with open(f) as f:
        offset = f.readline()
    offset = [[0,0]] + list(map(list , eval(offset) ))
    print(offset)
    output_shape, off_matrix, offset = calculate_offset( offset, img_np[0].shape[:2] )
    print( "img_np len : ", len(img_np) )
    print( "off_matrix len : ", len(off_matrix) )
    print(offset)
    for i in range(len(img_np)-1):
        w = img_np[i].shape[1]
        X = Scissors( offset[i][0], offset[i+1][0], w )
        img_np[i] = img_np[i][:,:X,:]

    img_width = [ img.shape[1] for img in img_np ]
    img_war = transform.warp( img_np[0], off_matrix[0], output_shape = output_shape, cval = -1 )
    linear_blending( img_war, img_width, offset, 0 )
    merge = np.where( img_war < 0, 0, img_war )
    
    for i in range( 1, pic_nums ) :
        print( f"warping img{i}" )
        img_war = transform.warp( img_np[i], off_matrix[i], output_shape = output_shape, cval = -1 )
        linear_blending( img_war, img_width, offset, i )
        img_war = np.where( img_war < 0, 0, img_war )
        merge += img_war

    merge_img = merge
    io.imsave( args.photo_dir / "linear_out.JPG", merge_img[:,:,0:3] )
    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--device",
        type=int,
        help="cuda device.",
        default=0,
    )
    parser.add_argument(
        "--photo_dir",  
        type=Path,
        help="directory to the photos",
        default="../data/p13_cylinder",
    )
    parser.add_argument(
        "--photo_nums",
        type=int,
        help="photo nums",
        default=26
    )
    parser.add_argument(
        "--offset",
        type=str,
        help="offset",
        default="../data/p13_cylinder/offset.normalize"
    )
    args = parser.parse_args()
    return args
if __name__ == "__main__" :
    args = parse_args()
    main(args)
