import cv2
import numpy as np
import MTB
def recover_offset(img, offset):
    print(f"img with offset {offset}...")
    ans = np.array(img.shape,dtype="float")
    W,H,RGB = img.shape
    for i in range(W):
        for j in range(H):
            for x in range(RGB):
                if i + offset[0] >= 0 and i + offset[0] < W \
                        and j + offset[1] >= 0 and j + offset[1] < H:
                    ans[i, j, x] = img[i+offset[0], j+offset[1], x]
    return ans

def HDR(files, times):
    final = np.zeros(files[0].shape)
    W, H, RGB = final.shape 
    time2_sum = sum(map(lambda x : x*x ,times))
    for x in range(W):
        for y in range(H):
            for rgb in range(RGB):
                for file, time in zip(files, times):
                    final[x,y,rgb] += file[x,y,rgb]*time
                final[x,y,rgb] /= time2_sum
    return final


if __name__ == "__main__":
    files = []
    print('Please put the file under the folder "raw_image" !')

    print('Give the list of file name separated with "," (ex: fileA.ppm, fileB.ppm, fileC.ppm, fileD.ppm):')
    files_name = input().split(',')
    files_name = [ file.strip() for file in files_name ]
    
    print('Give the list of shutter speed separated with "," (ex: 1, 0.5, 0.25, 0.125):')
    times = list(map(eval, input().split(',')))

    print(files_name)
    print(times)

    offset = MTB.MTB(files_name)
    X0, X1, Y0, Y1 = [0, 0, 0, 0]

    for file_name in files_name:
        print(f"reading image{file_name}....")
        img = cv2.imread(f"./raw_image/{file_name}").astype(np.float32)
        files.append(img)
    
    print(f"recover offset....")
    for x,y in offset:
        if X0 < x: X0 = x
        if X1 > x: X1 = x
        if Y0 < y: Y0 = y
        if Y1 > y: Y1 = y
    X1 += files[0].shape[0]
    Y1 += files[0].shape[1]

    for idx in range(len(files)):
        x,y = offset[idx]
        files[idx] = files[idx][X0-x:X1-x,Y0-y:Y1-y]
    
    print("processing HDR image")    
    final_img = HDR(files, times)
    cv2.imwrite(f"./image.hdr", final_img)
