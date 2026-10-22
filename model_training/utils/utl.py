from scipy.io import loadmat, matlab
import numpy as np
# loading .mat arrays (function from the internet)
def sec_to_hour(d) : 
    """
    translate a duration d (int or float) into a string in the format hour,minutes
    """
    h = int(d/3600)
    m = int( ((d/3600 - h)%1)*60  )
    return f"{h}h{m}mn"


def load_mat(filename):
    """
    This function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    def _check_vars(d):
        """
        Checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _toarray(d[key])
        return d

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _toarray(elem)
            else:
                d[strg] = elem
        return d

    def _toarray(ndarray):
        """
        A recursive function which constructs ndarray from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        if ndarray.dtype != 'float64':
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, matlab.mio5_params.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_toarray(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return np.array(elem_list)
        else:
            return ndarray

    
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_vars(data)

import torch
# Global Field Power (GFP) scaling
def gfp_scaling(M, j_pred, G): 
    # M: ground truth EEG data
    # j_pred: estimated source distibution, unscaled
    # G: leadfield matrix

    M_pred = G @ j_pred

    # Vectorized over all time instants (same maths as the per-timestep loop it replaces):
    # scale_t = std(M[:, t]) / std(M_pred[:, t]), with std(M_pred[:, t]) == 0 replaced by 1.
    std_pred = torch.std(M_pred, dim=0)
    std_pred = torch.where(std_pred == 0, torch.ones_like(std_pred), std_pred)
    j_pred_scaled = j_pred * (torch.std(M, dim=0) / std_pred)

    return j_pred_scaled


# patch 
def get_patch(order, idx, neighbors): 
    new_idx = np.array( [idx], dtype=np.int64 )
    #print(new_idx)

    if order == 0: 
        return new_idx
    
    else: 
        # for each order, find roder one neighbors of the current sources in patch
        for _ in range(order): 
            neighb = np.unique( neighbors[new_idx,:] )
            #neighb = neighb[~np.isnan(neighb)].astype(np.int64)
            neighb = neighb[neighb>0].astype(np.int64)
            #neighb = np.array(neighb, dtype=np.int64)

            #print(f"neighbors: {neighb}")
            new_idx = np.append( new_idx, neighb )
            #print(f"new indices: {new_idx}")
            
        
        return np.unique(new_idx)


import os
def prepare_results_folders( results_path:str, dataset_name:str,  expe_folder:str, sub_folders:list = ["traind_models", "figs", "eval", "logs"] ): 
    """ 
    Prepare the arborescence of folders to save results 
    - results_path: path to the folder in which to save results
    - dataset_name: name of the dataset used for the experiments
    - sub_folders: list of subfolders to create
    - expe_folder: name of the folder of a given experiment
    """

    results_path = f"{results_path}/{dataset_name}"


    os.makedirs(results_path, exist_ok=True)

    for sf in sub_folders: 
        os.makedirs(f"{results_path}/{sf}/{expe_folder}", exist_ok=True)

    return results_path, expe_folder


############### compute neighbors v2 
def get_neighbors(tris, verts): 
    n_verts = len(verts[0]) + len(verts[1])

    neighbors = [list() for _ in range(n_verts)]

    for hem in range(2): 
        i = 0
        idx_tris_old = np.sort(np.unique(tris[hem])).astype(np.int64)
        idx_vert_old = np.sort(np.unique(verts[hem])).astype(np.int64)

        missing_verts = np.setdiff1d(idx_tris_old, idx_vert_old)
        #idx_tris_new = np.arange(0, len(idx_tris_old))
        idx_vert_new = np.arange(0, len(idx_vert_old))

        vertices_lin = np.zeros((idx_vert_old.max()+1,1))
        vertices_lin[idx_vert_old,0] = idx_vert_new
        vertices_lin = vertices_lin.astype(np.int64)

        for v in verts[hem]: 
            triangles_of_v = np.squeeze(tris[hem] == v)
            triangles_of_v = np.squeeze(tris[hem][np.sum(triangles_of_v, axis=1) > 0])

            neighbors_of_v = np.unique(triangles_of_v)
            neighbors_of_v = neighbors_of_v[neighbors_of_v != v]
            neighbors_of_v = np.setdiff1d(neighbors_of_v, missing_verts)   
            

            #print(f"vert : {v}, {len(vertices_lin[neighbors_of_v,0])}")
            neighbors[i] = list( vertices_lin[neighbors_of_v,0] )
            i += 1

    l_max           = np.amax( np.array([len(l) for l in neighbors]) )
    neighb_array    = np.zeros( (len(neighbors), l_max) )
    for i in range(len(neighbors) ) : 
        l = neighbors[i]
        neighb_array[i,:len(l)] = l
        if len(l)<l_max: 
            neighb_array[i,len(l):] = None 

    return neighb_array.astype(np.int64)


from torch import nn
class logMSE(nn.Module): 
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, x_hat) : 
        mse = nn.MSELoss()
        return torch.log10( mse( x, x_hat ) )
    
class CosineSimilarityLoss(nn.Module):
    """
    Cosine similarity loss based on the cosine similarity function

    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, x_hat):
        cossim = nn.CosineSimilarity()
        cossim_val = -cossim(x, x_hat)
        return cossim_val.mean()