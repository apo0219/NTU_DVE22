from math import inf
import numpy as np
import cupy as cp
from Feature import Feature as feature
import pickle
from argparse import ArgumentParser, Namespace
import time
import copy
import random
from pathlib import Path
import math

dis = cp.ElementwiseKernel(
    'T a, T b',
    'T z',
    '''
    T trace = a - b;
    z = trace * trace;
    ''',
    'dis'
)

class bins:
    @staticmethod
    def dis(v1,v2):
        f = lambda a, b: (a-b)**2
        return cp.sum( f(v1,v2) )

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--device",
        type=int,
        help="cuda device.",
        default=0,
    )
    parser.add_argument(
        "--feature_path",
        type=Path,
        help="feature path",
        default = "../data/p13_cylinder"
    )
    args = parser.parse_args()
    return args

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


def main(args):
    cp.cuda.Device( args.device ).use()
    file_name = args.feature_path / "feature.pkl"
    
    with open(file_name,'rb') as f:
        L = pickle.load(f)
 
    for photo in L:
        for scale in photo:
            for feature in scale:
                normalize(feature)

    offset = []
    threshold = 0.2
    cnt = 0
    for A, B in zip(L, L[1:]):
        p = []
        print(f"processing pic {cnt} {cnt+1}")
        cnt += 1
        for L0, L1  in zip(A, B):  
            #print(len(L0), len(L1))
            features1 = cp.asarray( [f.vector for f in L0])
            features2 = cp.asarray( [f.vector for f in L1])
            
            suc = 0
            for idx, f2 in enumerate(features2):
                tmp = [ (cp.sum(dis(F,f2)), i) for i, F in enumerate(features1)]
                ans = min(tmp)
                if ans[0] < threshold :
                    p.append( v_sub(L0[ ans[1] ].loc_xys, L1[ idx ].loc_xys) )

        k = 500
        offset.append(RANSAC(k, 10, p))
    
    f = open( args.feature_path / "offset.normalize", 'w')
    print(file_name)
    f.write(str(offset))
    print(offset) 

if __name__ == "__main__" :
    args = parse_args()
    main(args)


