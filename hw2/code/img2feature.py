from argparse import ArgumentParser, Namespace
import math
from pathlib import Path
import pickle
import cupy as cp
from skimage import io
from cucim.skimage import filters, color, transform, feature, util
from Feature import Feature

corner_sc = cp.ElementwiseKernel(
    'T x2, T y2, T xy',
    'T z',
    '''
    T trace = x2 + y2;
    if ( trace == 0 ) {
        z = 0;
    } else {
        T det = x2 * y2 - xy * xy;
        z = det / trace;
    }
    ''',
    'score_for_corner'
)
def score( img ) :
    img = filters.gaussian( img, sigma = 1 )
    dy = filters.prewitt_h( img )
    dx = filters.prewitt_v( img )
    dx2 = filters.gaussian( dx * dx, sigma = 1.5 )
    dy2 = filters.gaussian( dy * dy, sigma = 1.5 )
    dxy = filters.gaussian( dx * dy, sigma = 1.5 )  
    return corner_sc( dx2, dy2, dxy )

def scale_tuple( t, n ) :
    return ( t[0] * n, t[1] * n )

def img_read_np_cp( pic_num, scale_num, path ):
    img_cp = []
    for i in range( pic_num ):
        file_name = path / f"img{'{0:02d}'.format(i)}.JPG"
        img = io.imread( file_name )
        print( "read : ", file_name )
        org_gpu = cp.asarray( img )
        print( "device : ", org_gpu.device ) 
        images_gpu = []
        images_gpu.append( color.rgb2gray( org_gpu ) * 255 )
        for j in range( 1, scale_num ):
            img_gsn = filters.gaussian( images_gpu[j - 1], sigma=1 )
            img_nxt = transform.resize( img_gsn, scale_tuple( img_gsn.shape, 0.5 ) )
            images_gpu.append( img_nxt )
        img_cp.append( images_gpu )
    return img_cp

def get_angle( dx, dy ):
    length = math.sqrt( dx * dx + dy * dy )
    cos = dx / length
    sin = dy / length
    return math.atan( sin / cos ) * 180 / math.pi

def save_object( obj, filename ):
    with open( filename, 'wb' ) as outp:
        pickle.dump( obj, outp, pickle.HIGHEST_PROTOCOL )

def main(args):
    pic_num = args.photo_nums
    scale_num = 4
    cp.cuda.Device( args.device ).use()
    img_cp = img_read_np_cp( pic_num, scale_num, args.photo_dir )
    all_feature = []
    for i in range( pic_num ):
        feature_for_pic = []
        for j in range( scale_num ):
            feature_with_specific_scale = []
            print( f"pic_{i}_{j} preprocess" )
            print( f"size : {img_cp[i][j].shape}")
            images_sc = score( img_cp[i][j] )
            corner_cor = feature.peak_local_max( images_sc, threshold_abs = 10, num_peaks = 1000, min_distance = max(1, int( img_cp[i][j].shape[1] / 80 ) ) ).get()
            blur = filters.gaussian( img_cp[i][j], sigma=4.5 )
            get_feature_from = filters.gaussian( img_cp[i][j], sigma=2 )
            dx = filters.prewitt_h( blur )
            dy = filters.prewitt_v( blur )
            window_size = 30
            h, w = img_cp[i][j].shape
            print( f"{len(corner_cor)} of features" )
            for p in corner_cor:
                row_lb = p[0] - window_size
                row_ub = h - p[0] - window_size
                column_lb = p[1] - window_size
                column_ub = w - p[1] - window_size
                if ( row_lb < 0 or row_ub < 0 or column_lb < 0 or column_ub < 0 ): continue
                cut = util.crop( get_feature_from, ( ( row_lb, row_ub ), ( column_lb, column_ub ) ), copy = True )
                ang = get_angle( dx[p[0]][p[1]], dy[p[0]][p[1]] )
                rot = transform.rotate( cut, angle = -ang )
                cut_rot = util.crop( rot, ( ( window_size - 20, 60 - window_size - 20 ), ( window_size - 20, 60 - window_size - 20 ) ), copy = True )
                resize = transform.resize( cut_rot, ( 8, 8 ) ).get()
                feature_with_specific_scale.append( Feature( ( p[1], p[0], j ), resize ) )
            feature_for_pic.append( feature_with_specific_scale )
        all_feature.append( feature_for_pic )
    save_object( all_feature, args.photo_dir / 'feature.pkl' )

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
    args = parser.parse_args()
    return args
if __name__ == "__main__" :
    args = parse_args()
    main(args)
