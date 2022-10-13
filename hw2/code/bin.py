from math import inf
import numpy as np
from Feature import Feature as feature
import pickle
from pathlib import Path
from skimage import io
import math
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
import random

class bins:
    table = [[], [], []]
    n = 10
    _bin = [ [ [ [] for i in range(10) ] for j in range(10) ] for k in range(10) ]
    
    @classmethod
    def get_index(cls,V):
        ans = []
        for n in range(3):
            for i in range(cls.n):
                #print(i,cls.table[n][i])
                if V[n] < cls.table[n][i]:
                    ans.append(i)
                    break
        return ans
    @staticmethod
    def dis(v1,v2):
        f = lambda a, b: (a-b)**2
        vfunc = np.vectorize(f)
        return np.sum( vfunc(v1,v2) )
    
    @classmethod
    def find(cls, F):
        x,y,z = cls.get_index( F.mask )
        dir = []
        for i in range(-1,1):
            for j in range(-1,1):
                for k in range(-1,1):
                    dir.append((i,j,k))
        
        best, score = None, inf
        for _x, _y, _z in dir:
            if x+_x < 0 or x+_x >= cls.n: continue
            if y+_y < 0 or y+_y >= cls.n: continue
            if z+_z < 0 or z+_z >= cls.n: continue
                
            for f in cls._bin[x+_x][y+_y][z+_z]:
                D = cls.dis(F.vector,f.vector)
                if D < score:
                    best, score = f, D
        
        return best, score

def v_sub(A, B):
    return (A[0]-B[0]) * pow(2,A[2]) , (A[1]-B[1]) * pow(2,A[2])

def v_dis(A, B):
    return (A[0]-B[0])**2 + (A[1]-B[1])**2 

def RANSAC(k, D, L): 
    ans, best = 0, 0
    for _ in range(k):
        c, n = random.choice(L), 0
        for p in L:
            if v_dis(c, p) < D:
                n += 1
        if n > best:
            ans = c
            best = n
    return ans

def normalize(feature):
    mu = np.sum(feature.vector)/64
    sigma = 0
    for x in feature.vector:
        for y in x:
            sigma += y*y

    sigma -= mu*mu
    feature.vector /= math.sqrt(sigma)
    
    mu = np.sum(feature.vector)/64
    feature.vector -= mu

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--feature_path",
        type=Path,
        help="feature path",
        default = "../data/p13_cylinder"
    )
    args = parser.parse_args()
    return args

def main(args):
    file_name = args.feature_path / "feature.pkl"

    with open(file_name,'rb') as f:
        L = pickle.load(f)

    for photo in L:
        for scale in photo:
            for feature in scale:
                normalize(feature)

    cnt = 0
    threshold = 0.5
    offset = []
    
    for L0, L1 in zip(L, L[1:]):
        p = []
        print(f"processing pic {cnt} {cnt+1}")
        cnt += 1
        
        for A, B in zip(L0,L1):
            N = len(A)
            l, L, i = N/10, [], 0
            while i < N:  i+= l; L.append(int(i))
            
            for n in range(3):
                A.sort(key =  lambda x : x.mask[n])
                for i in L[:-1]:
                    bins.table[n].append(A[i].mask[n])
                bins.table[n].append(inf)
            
            for F in A:
                x,y,z = bins.get_index( F.mask )
                bins._bin[x][y][z].append(F)
    
            for idx, f2 in enumerate(B):
                ans = bins.find( f2 )
                if ans[1] < threshold :
                    p.append( v_sub( ans[0].loc_xys, f2.loc_xys) )

        k = 500
        offset.append(RANSAC(k, 10, p))

    f = open( args.feature_path / "offset.normalize.bin", 'w')
    print(file_name)
    f.write(str(offset))
    print(offset)

if __name__ == "__main__" :
    args = parse_args()
    main(args)
