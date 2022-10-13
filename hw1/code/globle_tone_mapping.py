import cv2
import numpy as np
import math

def globle( img, alpha, beta, L_w, g, r, b ):
    ( H, W, C ) = img.shape
    B, G, R = 0, 1, 2
    
    img_chw = np.transpose( img, ( 2, 0, 1 ) )

    L_bar = np.zeros( 3 )
    L_m = np.zeros( ( C, H, W ) )
    L_d = np.zeros( ( C, H, W ) )
    
    for i in range( 3 ):
        L_bar[i] = math.exp( np.log( img_chw[i] + 0.000001 ).sum() / (W * H) ) 
        L_m[i] = img_chw[i] * alpha / L_bar[i] + beta
        L_d[i] = L_m[i] * ( 1 + L_m[i] / math.pow( L_w , 2 ) ) / ( 1 + L_m[i] )

    L_d[G] = L_d[G] * g
    L_d[R] = L_d[R] * r
    L_d[B] = L_d[B] * b
    L_d = np.transpose( L_d, ( 1, 2, 0 ) )
    return L_d

def local( img, phi, alpha, epslon ):
    ( H, W, C ) = img.shape
    B, G, R = 0, 1, 2
    img_chw = np.transpose( img, ( 2, 0, 1 ) )
    L_blur = np.zeros( ( C, H, W ) )
    L_m = np.zeros( ( C, H, W ) )
    L_d = np.zeros( ( C, H, W ) )
    L_bar = np.zeros( 3 )
    def V( s, s_1, r ):
        return ( s - s_1 ) / ( math.pow( 2, phi ) * alpha / math.pow( r, 2 ) + s )
    
    def find_Lblur( c, x, y ):
        print( '(', c, x, y, ')' )
        L_s = img_chw[c][x][y]
        log_sum = 0
        count = 0
        for i in range( -1, 2 ):
            for j in range( -1, 2 ):        
                if ( x + j < 0 or x + j >= H ):
                    continue
                if ( y + i < 0 or y + i >= W ):
                    continue
                count += 1
                log_sum += math.log( img_chw[c][x + j][y + i] )
        L_s_1 = math.exp( log_sum / count )
        d = 1
        while ( abs( V( L_s, L_s_1, d ) ) < epslon ):
            if ( d > min( H // 2, W // 2 ) ):
                break
            print(d)
            d += 1
            L_s = L_s_1
            if ( x - d > 0 ):
                for i in range( max( 0, y - d ), min( y + d, W - 1 ) + 1 ):
                    count += 1
                    log_sum += math.log( img_chw[c][x - d][i] + 0.000001 )
            if ( x + d < H ):
                for i in range( max( 0, y - d ), min( y + d, W - 1 ) + 1 ):
                    count += 1
                    log_sum += math.log( img_chw[c][x + d][i] + 0.000001 )
            if ( y - d > 0 ):
                for i in range( max( 0, x - d + 1), min( x + d - 1, H - 1 ) + 1 ):
                    count += 1
                    log_sum += math.log( img_chw[c][i][y - d] + 0.000001 )
            if ( y + d < W ):
                for i in range( max( 0, x - d + 1), min( x + d - 1, H - 1 ) + 1 ):
                    count += 1
                    log_sum += math.log( img_chw[c][i][y + d] + 0.000001 )
            L_s_1 = math.exp( log_sum / count )
        return L_s

    for i in range( C ):
        for j in range( H ):
            for k in range( W ):
                L_blur[i][j][k] = find_Lblur( i, j, k )
    for i in range( C ):
        L_bar[i] = math.exp( np.log( img_chw[i] + 0.000001 ).sum() / (W * H) ) 
        L_m[i] = img_chw[i] * alpha / L_bar[i] 
        L_d[i] = L_m[i] / ( 1 + L_blur[i] )
    L_d = np.transpose( L_d, ( 1, 2, 0 ) )
    return L_d
         
def barrel_undistort(img, focal_x=18, focal_y=22.5):
    '''
    To fix the barrel distortion we use the cv2.undistort.
    Reference: https://stackoverflow.com/questions/26602981/correct-barrel-distortion-in-opencv-manually-without-chessboard-image
    '''
    distCoeff = np.zeros((4,1),np.float64)

    k1 = -1.0e-5 # negative to remove barrel distortion
    k2 = 0.0
    p1 = 0.0
    p2 = 0.0

    distCoeff[0,0] = k1
    distCoeff[1,0] = k2
    distCoeff[2,0] = p1
    distCoeff[3,0] = p2

    cam = np.eye(3,dtype=np.float32)

    cam[0,2] = W/2.0        # define center x
    cam[1,2] = H/2.0        # define center y
    cam[0,0] = focal_x      # define focal length x
    cam[1,1] = focal_y      # define focal length y
    
    return cv2.undistort(img,cam,distCoeff)

if __name__ == "__main__":

    file = input("Give the file name: ")
    
    img = cv2.imread( file, cv2.IMREAD_ANYDEPTH )
    print( type( img ) )
    
    ( H, W, C ) = img.shape
    B, G, R = 0, 1, 2

    img = globle( img, 0.09, 0.10, 1.5, 75/100, 130/100, 100/100 )
    img = barrel_undistort(img)
    cv2.imwrite( f"{file}.png" , img*255)
