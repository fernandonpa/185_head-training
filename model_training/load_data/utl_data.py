# utils to manipulate data
import os
import numpy as np
import torch 
import json 
import sys 
from scipy.io import loadmat
##########################
# Loading functions
##########################
# functions to load from a file
def load_eeg_data_from_file(file_name:str)->np.ndarray:
    """ 
    load eeg data (.mat) from a file name
    input: file_name = name of the .mat file to laod
    output: eeg data as numpy array (dim:n_electrodes,n_times,n_trials)
    """
    #print("Loading EEG data from mat file: ", file_name)
    return loadmat(f"{file_name}")['eeg_data']['EEG']

# for ource data... file + number of sources to reconstruct the data
# NB: if datasets are too big, possible to load only the active and noise signals as "sparse" matrix (signal and index), and
# to reconstruc the data inside the loops


def load_src_extended_data_from_file(file_name:str, md_file_name:str, n_sources:int, n_times:int)-> np.ndarray:
    """
    load source data from file name . Source data is stored in 2 files: active
    source data and noise source data. 
    @TODO: 12.07.2022: there is no noise data - add noise handling.
    Only signals from the active sources are stored, so to use it first reconstruct the whole data from 
    the signal, the total number of sources and the index/indices of active sources

    input
    - file_name : name of the .mat file
    - n_sources : number of sources of the source space (to reconstruct source data)

    output: 
    - src_data  : source data (dim: n_sources, n_times, n_trials)

    NB: if data to empty -> only load signals and index/indices and build the whole matrix data 
    only for evalutation of the model
    """
    src_data = np.zeros((n_sources, n_times))

    with open(md_file_name,"r") as f: 
        md_src  = json.load(f)
    seeds   = md_src["seeds"]
    act_src_idx = []

    # if only one patch
    if md_src["n_patch"]==1: 
        act_src_idx = md_src["act_src"][f"patch_1"]

    else: 
        for p in range(md_src["n_patch"]): 
            # if patch of order 0 (i.e single source)
            if md_src["orders"][p]>0: 
                act_src_idx += md_src["act_src"][f"patch_{p+1}"]
            
            # if patch of order > 0
            else: 
                act_src_idx.append( md_src["act_src"][f"patch_{p+1}"])
    

    # print("file_name: source ", file_name)
    src_data[act_src_idx, :] = loadmat(f"{file_name}")['Jact']['Jact'][0][0]
    
    return src_data #, act_src_idx

# option 1: load eeg data and source data separately
def load_eeg_data(js_eeg:dict, general_config_dict:dict, ids:list, as_tensor:bool=False):
    """  
    load eeg data from a liste of ids
    input: 
    - folders   : FolderStructure object containing the folder names of al data
    - js_eeg    : dictionary of eeg files information (ids and file names)
    - general_config_dict   : dictionary of general configuration information
    - ids   : ids of data to load

    output: 
    - eeg_data  : numpy array of eeg data, dimension (len(ids), n_electrodes, n_times)

    @TODO: add the case where multiple trials are simulated (n_trials>1)
    """
    n_electrodes    = general_config_dict['electrode_space']['n_electrodes']
    n_times         = general_config_dict['rec_info']['n_times']

    n_samples       = len(ids)
    eeg_data        = np.empty((n_samples, n_electrodes, n_times))

    i = 0
    for id in ids:
        #file_name = f"{folders.eeg_folder}/{js_eeg[id]['eeg_file']}"
        file_name   = f"{js_eeg[id]}"
        # print("file_name: ", file_name)
        eeg_data[i, :, :] = load_eeg_data_from_file(file_name)[0][0]
        i += 1

    if as_tensor: 
        eeg_data = torch.from_numpy(eeg_data)

    return eeg_data


def load_src_extended_data(js_md, js_src, general_config_dict, ids, as_tensor=False):
    """  
    load source data from a liste of ids
    input: 
    - folders               : FolderStructure object containing the folder names of al data
    - js_src                : dictionary of source files information 
                            (ids, file names, active source index/indices)
    - general_config_dict   : dictionary of general configuration information
    - ids                   : ids of data to load

    output: 
    - src_data  : numpy array of source data, dimension (len(ids), n_sources, n_times)

    @TODO: add the case where multiple trials are simulated (n_trials>1)
    """
    n_sources   = general_config_dict['source_space']['n_sources']
    n_times     = general_config_dict['rec_info']['n_times']

    n_samples   = len(ids)
    src_data    = np.empty((n_samples, n_sources, n_times))

    i = 0
    for id in ids:
        #file_name           = f"{folders.active_source_folder}/{js_src[id]['act_src_file_name']}"
        file_name    = f"{js_src[id]}"
        md_file_name = js_md[id]
        act_src_idx         = js_src[id]
        src_data[i, :, :]   = load_src_extended_data_from_file(file_name, md_file_name, n_sources, n_times)
        i += 1

    if as_tensor: 
        src_data = torch.from_numpy(src_data)
    return src_data


import colorednoise as cn
def add_noise_snr(snr_db, signal, noise_type = {"white":1.}, return_noise=False): 
    """  
    Return a signal which is a linear combination of signal and noise with a ponderation to have a given snr
    noise_type : dict key = type of noise, value = ponderation of the noise (for example use white and pink noise)
    """

    snr     = 10**(snr_db/10)
    noise = 0
    dims = [signal.shape[i] for i in range(len(signal.shape))]
    for n_type, pond in noise_type.items() : 
        if n_type=="white": 
            noise = noise + pond * np.random.randn( *dims )
        if n_type=="pink" : 
            beta = (1.8-0.2)*np.random.rand(1) + 0.2
            noise = noise + pond * cn.powerlaw_psd_gaussian(beta, signal.shape)

    if len(signal.shape)==2: 
        # 2D signal
        x = signal + (noise/np.linalg.norm(noise))*(np.linalg.norm(signal)/np.sqrt(snr))
    elif len(signal.shape)==3:
        # batch data 
        noise_norm = np.expand_dims( np.linalg.norm(noise, axis=(1,2)), (1,2) )
        sig_norm = np.expand_dims( np.linalg.norm(signal, axis=(1,2)), (1,2) )
        x = signal + (noise/noise_norm)*(sig_norm/np.sqrt(snr))
        #x = signal + (noise/np.linalg.norm(noise, axis=(1,2)))*(np.linalg.norm(signal,axis=(1,2))/np.sqrt(snr))
    else: 
        x = None
        sys.exit('Signal must be of dimension 2 (unbatched data) or 3 (batched data)')
        
    if return_noise : 
        return x, noise
    else: 
        return x

def tensor_range_scaling(x, inf, sup): 
    """
    rescale x so that it max(x) == sup and min(x) == inf.
    !! works for x a tensor of a single tensor (not a batch)
    returns rescaled_x, the rescaled tensor
    """
    x_maxs = torch.max(x)
    x_mins = torch.min(x)

    scale_factor = (sup - inf) / (x_maxs - x_mins) 
    rescaled_x = (x - x_mins) * scale_factor + x_mins
    return rescaled_x

def array_range_scaling(x, inf, sup): 
    """
    rescale x so that it max(x) == sup and min(x) == inf.
    !! works for x a tensor of a single tensor (not a batch)
    returns rescaled_x, the rescaled tensor
    """
    x_maxs = np.max(x)
    x_mins = np.min(x)

    scale_factor = (sup - inf) / (x_maxs - x_mins) 
    rescaled_x = (x - x_mins) * scale_factor + x_mins
    return rescaled_x
'''
def get_matching_info(data_folder_name, general_config_dict):
    """
    intput: 
    - folders   : FolderStrucure object containing all folder names
    - general_config_dict   : dictionary of general configuration information 
        of the simulation
    
    output: 
    - match_dict : json file of source and eeg names
    - ids_dict    : ids of the files (id of eeg file match id of the source data which was used to generate the eeg)
    """


    simu_name           = general_config_dict['simu_name']
    source_space_info   = general_config_dict['source_space']
    eeg_snr             = general_config_dict['eeg_snr']

    match_file = f"{data_folder_name}/{simu_name}/{simu_name}{source_space_info['src_sampling']}_match_json_file.json"

    # load json configuration file for eeg data
    with open(match_file, "r") as f:
        match_dict = json.load(f)

    
    # get the list of ids of the different files
    ids = list(match_dict.keys())

    eeg_dict = {
        f"{ids[k]}": match_dict[f"{ids[k]}"][f"eeg_file_name"] for k in range(len(ids))
    }


    src_dict = {
        f"{ids[k]}": match_dict[f"{ids[k]}"][f"act_src_file_name"] for k in range(len(ids))
    }

    md_dict = {
        f"{ids[k]}": match_dict[f"{ids[k]}"][f"md_json_file_name"] for k in range(len(ids))
    }
    return ids, eeg_dict, src_dict, md_dict
'''

def get_matching_info(data_folder_name, general_config_dict, root_simu):
    """
    intput:
    - folders   : FolderStrucure object containing all folder names
    - general_config_dict   : dictionary of general configuration information
        of the simulation
    
    output:
    - match_dict : json file of source and eeg names
    - ids_dict    : ids of the files (id of eeg file match id of the source data which was used to generate the eeg)
    """


    simu_name = general_config_dict["simu_name"]
    source_space_info = general_config_dict["source_space"]

    # Typical location (generated by the simulation pipeline):
    #   <data_folder_name>/<simu_name>/<simu_name><src_sampling>_match_json_file.json
    # Some users keep a copy at:
    #   <root_simu>/<simu_name>/<simu_name><src_sampling>_match_json_file.json
    expected_1 = os.path.normpath(
        os.path.join(
            data_folder_name,
            simu_name,
            f"{simu_name}{source_space_info['src_sampling']}_match_json_file.json",
        )
    )
    expected_2 = os.path.normpath(
        os.path.join(
            root_simu,
            simu_name,
            f"{simu_name}{source_space_info['src_sampling']}_match_json_file.json",
        )
    )

    if os.path.isfile(expected_1):
        match_file = expected_1
    elif os.path.isfile(expected_2):
        match_file = expected_2
    else:
        raise FileNotFoundError(
            "Could not find match JSON file. Tried:\n"
            f"- {expected_1}\n"
            f"- {expected_2}\n"
        )

    with open(match_file, "r") as f:
        match_dict = json.load(f)
    # get the list of ids of the different files
    ids = list(match_dict.keys())
    def _remap_if_needed(path_str: str) -> str:
        """
        The match JSON may contain absolute paths from another machine (e.g. C:\\Users\\...).
        If the file exists as-is, keep it. Otherwise, try to remap it to `root_simu` by
        preserving the tail starting from 'constrained'/'unconstrained'.
        """
        if not isinstance(path_str, str):
            return path_str

        # normalize separators for robust substring checks
        p = os.path.normpath(path_str)
        if os.path.exists(p):
            return p

        # Try remapping from orientation marker
        p_fwd = path_str.replace("\\", "/")
        root = root_simu
        for ori in ("constrained", "unconstrained"):
            marker = f"/{ori}/"
            if marker in p_fwd:
                tail = p_fwd.split(marker, 1)[1]
                candidate = os.path.normpath(os.path.join(root, ori, *tail.split("/")))
                return candidate

        # Fallback: return normalized path (even if missing) for clearer error messages downstream
        return p

    eeg_dict = {}
    src_dict = {}
    md_dict = {}
    for k in range(len(ids)):
        eeg_file_name = match_dict[f"{ids[k]}"]["eeg_file_name"]
        act_src_file_name = match_dict[f"{ids[k]}"]["act_src_file_name"]
        md_file_name = match_dict[f"{ids[k]}"]["md_json_file_name"]

        eeg_dict[f"{ids[k]}"] = _remap_if_needed(eeg_file_name)
        src_dict[f"{ids[k]}"] = _remap_if_needed(act_src_file_name)
        md_dict[f"{ids[k]}"] = _remap_if_needed(md_file_name)

    return ids, eeg_dict, src_dict, md_dict

def replace_root(root_simu, x, offset=6, split_char = '/') :
    splitted = x.split(split_char)
    splitted[:offset] = []
    new_str = root_simu
    for s in splitted :
        new_str += split_char + s
    return new_str