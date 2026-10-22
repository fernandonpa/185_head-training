import numpy as np
import matplotlib.pyplot as plt

def get_component_extended_src(order, seed, neighbors, spos, erp_params, erp_dev, timeline):

    amplitude = erp_params['ampl']
    patch = get_patch(order, seed, neighbors)
    n_source_in_patch = patch.shape[0]

    c = []
    if order > 0 : 
        seed_pos = spos[seed,:]
        d_in_patch = np.sqrt(
            np.sum(
            (seed_pos - spos[patch,:])**2,1
            )
        )
        #print(f"size patch {patch.shape}, dist size : {d_in_patch.shape}")
        patch_dim = np.max(d_in_patch)
        #d_in_patch = d_in_patch/patch_dim
        #print(f"patch_dim {patch_dim}")
        #sig = patch_dim / np.sqrt(2*np.log10(2)) #/2
        sig = np.max(d_in_patch)/2
        #print(f"sigma dist : {sig}")
        ampl = amplitude * np.exp(-0.5*(d_in_patch/sig)**2)
        
        for s in range(n_source_in_patch):

            tmp_c = erp_component(
                source=patch[s], 
                erp_params= {'ampl':ampl[s], 'width':erp_params['width'], 'center':erp_params['center']}, 
                erp_dev=erp_dev, 
                timeline=timeline)

            c.append(tmp_c)

    else : 
        c.append( erp_component(patch[0], erp_params, erp_dev, timeline) )
    return c, patch, patch_dim


class erp():
    def __init__(self, source, params, timeline) -> None:
        self.source = source
        self.params = params
        self.timeline = timeline
    
    def signal(self) :
        t_vec = np.arange(
            0, self.timeline['length']*1e-3, 1/self.timeline['srate']
        )
        #t_vec = np.linspace(
        #    0,
        #    (self.timeline['length']/1e3*self.timeline['srate'] - 1)/self.timeline['srate'],
        #     int(self.timeline['length']/1e3*self.timeline['srate']) )
        #print(t_vec.shape)
        # self.time = t_vec
        #print(f"ampl : {self.params['ampl']}, center : {self.params['center']}, width : {self.params['width']}")
        center = self.params['center']*1e-3
        sgm = self.params['width']*1e-3 / 6
        signl = self.params['ampl'] * np.exp(-0.5*((t_vec - center)/sgm)**2)

        return t_vec, signl  
    
def erp_component( source, erp_params, erp_dev, timeline ):
    # todo : take erp_dev into account, for now we will do as if deviation parameters were
    # taken into account elsewhere in the code
    return erp(source=source, params=erp_params, timeline=timeline)

def generate_scalp_data(c_tot, leadfield, timeline):
    n_times = int(timeline['length']*1e-3*timeline['srate'])
    #scalp_data = np.zeros((leadfield.shape[0], timeline['length']*1e-3*timeline['srate']))
    act_src_idx = [c.source for c in c_tot]
    act_leadfield = leadfield[:, act_src_idx]
    act_src = np.zeros((len(act_src_idx), n_times))
    for i in range(len(c_tot)) : 
        _, act_src[i,:] = c_tot[i].signal()

    scalp_data = act_leadfield @ act_src
    return scalp_data, act_src



def get_patch(order, idx, neighbors): 
    new_idx = np.array( [idx], dtype=int )
    #print(new_idx)

    if order == 0: 
        return new_idx
    
    else: 
        # for each order, find roder one neighbors of the current sources in patch
        for _ in range(order): 
            neighb = np.unique( neighbors[new_idx,:] )
            #neighb = neighb[~np.isnan(neighb)].astype(np.int64)
            neighb = neighb[neighb>0]#.astype(int)
            neighb = np.array(neighb, dtype=np.int64)

            #print(f"neighbors: {neighb}")
            new_idx = np.append( new_idx, neighb )
            #print(f"new indices: {new_idx}")
            
        
        return np.unique(new_idx)
    

def get_neighbors(tris, verts): 
    n_verts = len(verts[0]) + len(verts[1])

    neighbors = [list() for _ in range(n_verts)]
    i = 0
    for hem in range(2): 
        
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
            if hem == 1:
                neighbors[i] = [k + len(verts[0]) for k in neighbors[i]]
            i += 1

    l_max           = np.amax( np.array([len(l) for l in neighbors]) )
    neighb_array    = np.zeros( (len(neighbors), l_max) )
    for i in range(len(neighbors) ) : 
        l = neighbors[i]
        neighb_array[i,:len(l)] = l
        if len(l)<l_max: 
            neighb_array[i,len(l):] = None 

    return neighb_array.astype(np.int64)