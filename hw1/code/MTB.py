import cv2
import numpy as np

def image_GrayScale(img):
    B, G, R = 0, 1, 2
    (Weight, Height, BGR) = img.shape
    Y = lambda pixel: (54*pixel[R]+183*pixel[G]+19*pixel[B])/256
    Gray_Scale_img = np.zeros((Weight, Height))
    for i in range(Weight):
        for j in range(Height):
            Gray_Scale_img[i][j] = Y(img[i][j])
    
    return Gray_Scale_img

def Binary(Gray_Scale_img):
    avg = np.mean(Gray_Scale_img)
    Max = np.max(Gray_Scale_img)
    Min = np.min(Gray_Scale_img)
    step = (Max-Min)/20
    Binary_img = np.where(Gray_Scale_img > avg, 0, 1)
    Mask_img_G = np.where( Gray_Scale_img > (avg+step), 1, 0)
    Mask_img_L = np.where( Gray_Scale_img < (avg-step), 1, 0)
    Mask_img = Mask_img_L + Mask_img_G

    return Binary_img, Mask_img

def compress(img, depth=7):
    img_List = []
    img_List.append( Binary(img) )
    last = img
    for _ in range(1,depth):
        last_w, last_h = last.shape
        w, h = last_w//2, last_h//2
        now_img = np.zeros((w, h))
        for i in range(w):
             for j in range(h):
                now_img[i][j] =(  last[ 2*i ][ 2*j ] \
                                + last[2*i+1][ 2*j ] \
                                + last[ 2*i ][2*j+1] \
                                + last[2*i+1][2*j+1] )/4 
        last = now_img
        img_List.append( Binary(now_img) )
    return img_List[::-1]

def get_offset(img0, img1, mask, offset):
    direction = [ (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    for i in range(9):
        direction[i] = (offset[0] + direction[i][0], offset[1] + direction[i][1])
    
    w, h = img0.shape
    score = [0]*9
    
    for i in range(1,w):
        for j in range(1,h):
            if i + direction[0][0] < 0 or j + direction[0][1] < 0: continue
            if i + direction[8][0] >= w or j + direction[8][1] >= h: continue
            for k in range(9):
                score[k] += (img0[i,j] ^ img1[i+direction[k][0], j+direction[k][1]]) & mask[i,j]

    for i in range(9):
        if score[i] == min(score):
            ans = direction[i]
            break
    return ans

def cmp_img(a,b):
    w, h = a.shape
    for i in range(w):
        for j in range(h):
            if a[i,j] != b[i,j]: return False
    return True

def MTB(files_name): 
    print("compress first file....")
    print(f"./raw_image/{files_name[0]}")
    img = cv2.imread(f"./raw_image/{files_name[0]}")
    img = image_GrayScale(img)
    file0 = compress(img, 7)
    offsets = []
    for file in files_name:
        print(f"processing image {file}....")
        img = cv2.imread(f"./raw_image/{file}")
        img = image_GrayScale(img)
        file1 = compress(img, 7)
        offset = [0, 0]
        for f0, f1 in zip(file0,file1): 
            img0, mask = f0
            img1 = f1[0]
            offset[0] *= 2
            offset[1] *= 2
            tmp = get_offset(img0, img1, mask, offset)
            offset[0] = tmp[0]
            offset[1] = tmp[1] 
        offsets.append(offset)
    return offsets

