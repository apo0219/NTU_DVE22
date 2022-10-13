import numpy as np
class Feature:
    mask0 = np.ones((8,8)); mask0[:4,:] = -1
    mask1 = np.ones((8,8)); mask1[:,:4] = -1
    mask2 = np.ones((8,8)); mask2[:4,:4] = -1; mask2[4:,4:] = -1
    
    @staticmethod
    def standardize(vector):
        sigma = np.var(vector)
        avg = np.average(vector)
        vector = (vector - avg) / sigma
        return vector    
    
    def __init__(self, loc_tup, vector):
        self.loc_xys = loc_tup
        self.vector = self.standardize(vector)
        self.mask = []
        self.mask.append( np.sum( self.vector * self.mask0 ) )
        self.mask.append( np.sum( self.vector * self.mask1 ) )
        self.mask.append( np.sum( self.vector * self.mask2 ) )

    def __repr__(self):
        return str(self.vector)


