from skimage import io, util
from pathlib import Path
from argparse import ArgumentParser, Namespace
import numpy as np
def convert_xy( x, y ):
    global center, f

    new_x = ( f * np.tan( (x - center[0]) / f ) ) + center[0]
    new_y = ( (y - center[1]) / np.cos( (x - center[0]) / f ) ) + center[1]
    
    return new_x, new_y

def main(args):
    global center, f
    for i in range( args.photo_nums ):
        file_name = args.photo_dir / f"img{'{0:02d}'.format(i)}.JPG"
        img = io.imread( file_name )
        print( "read : ", file_name )
        h, w = img.shape[:2]
        center = [ w // 2, h // 2 ]
        f = 19 * 2664 / 4.55
        new_img = np.zeros( img.shape, dtype=np.uint8 )
        new_xy = np.array( [ np.array([i, j]) for i in range(w) for j in range(h) ] )
        new_x = new_xy[:, 0]
        new_y = new_xy[:, 1]
        org_x_flt, org_y_flt = convert_xy( new_x, new_y )
        org_x_int = org_x_flt.astype(int)
        org_y_int = org_y_flt.astype(int)

        valid_mask = ( org_x_int >= 0 ) * ( org_x_int <= w-2 ) * ( org_y_int >= 0 ) * ( org_y_int <= h-2 ) 
        new_x = new_x[valid_mask]
        new_y = new_y[valid_mask]
        org_x_flt = org_x_flt[valid_mask]
        org_y_flt = org_y_flt[valid_mask]
        org_x_int = org_x_int[valid_mask]
        org_y_int = org_y_int[valid_mask]

        dx = org_x_flt - org_x_int
        dy = org_y_flt - org_y_int

        weight_tl = ( 1.0 - dx ) * ( 1.0 - dy )
        weight_tr = ( dx ) * ( 1.0 - dy )
        weight_bl = ( 1.0 - dx ) * ( dy )
        weight_br = ( dx ) * ( dy )
        new_img[ new_y, new_x, : ] = ( weight_tl[:, None] * img[org_y_int    , org_x_int    , :] ) +\
                                    ( weight_tr[:, None] * img[org_y_int    , org_x_int + 1, :] ) +\
                                    ( weight_bl[:, None] * img[org_y_int + 1, org_x_int    , :] ) +\
                                    ( weight_br[:, None] * img[org_y_int + 1, org_x_int + 1, :] )
        left_edge = 0
        for j in range(w):
            if ( new_img[h // 2][j][0] == 0 ) :
                left_edge = j
                continue
            break
        right_edge = 0
        for j in range(w-1 , 0, -1):
            if ( new_img[h // 2][j][0] == 0 ) :
                right_edge += 1
                continue
            break
        new_img = util.crop( new_img, ( (0, 0), (left_edge, right_edge), (0, 0) ) )
        io.imsave( args.output_dir / f"img{'{0:02d}'.format(i)}.JPG", new_img[:,1:,:] )
        print( args.output_dir / f"img{'{0:02d}'.format(i)}.JPG save" )

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--photo_dir",  
        type=Path,
        help="directory to the photos",
        default="../data/p13",
    )
    parser.add_argument(
        "--photo_nums",
        type=int,
        help="photo nums",
        default=26
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="where to store output image",
        default = "../data/p13_cylinder"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__" :
    args = parse_args()
    main(args)
