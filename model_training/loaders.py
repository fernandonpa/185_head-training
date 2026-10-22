from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat, savemat
#import h5py
from utils.utils_deepsif import add_white_noise, ispadding
import random
try:
    import mne  # optional (used only for optional filtering)
    HAS_MNE = True
except ModuleNotFoundError:
    mne = None
    HAS_MNE = False
import json
from os.path import expanduser
import os

############# for sereega dataset
from load_data import utl_data
import torch


class EsiDatasetds_new(Dataset):
    """
    Dataset to handle data simulated with physical - sereega based model
    Input init :
        - config_file (str) :  name of the configuration file of the simulation
        - simu_name (str) : name of the simulation
        - source_space (obj: SourceSpace): source space corresponding to the dataset
        - electrode_montage (obj:ElectrodeMontage): electrode montage corresponding to the dataset
        - to_load (int) : number of samples to load i.e dataset length
        - snr_db (float or str): SNR of the EEG data (between EEG from source and additive noise)
        - noise_type (dict): type of noise to add to the data + scaling factor (ex : {"white":1., "pink": 1.})
        - norm (str): linear or max-max -> type of normalisation to use
    Attributes :
        - to_load : number of samples to load / dataset length
        - config_file : configuration file of the simulation
        - source_space : source space of the simulation
        - electrode_montage : electrode montage of the simulation
        - norm : type of normalisation to use (shoul by linear or max-max)
        - general_config_dict : configuration dictonnary (content of config_file)
        - ori : orientation of sources (constrained or unconstrained)
        - ids : ids of the loaded data
        - eeg_dict :
        - src_dict :
        - match_dict : dictionnary to match an EEG file to the source distribution file corresponding
        - max_eeg : maximum absolute value of each EEG sample (used for normalisation purposes)
        - max_src : maximum absolute value of each source sample (0 if norm == linear)
        - snr_db : SNR of the EEG data (can be int or "random")
        - noise_type (dict): type of noise + scaling factor
        - md_dict : dictionnary of "metedata" for each sample (md_dict[id] -> active sources, seeds, orders...)

    """

    def __init__(
        self,
        root_simu,
        config_file,
        simu_name,
        source_space,
        electrode_montage,
        to_load,
        snr_db,
        noise_type,
        norm="linear",
    ):
        super().__init__()
        self.to_load = to_load
        self.root_simu = root_simu

        self.config_file = config_file
        self.simu_name = simu_name
        self.source_space = source_space
        self.electrode_montage = electrode_montage
        self.norm = norm

        with open(config_file, "r") as f:
            self.general_config_dict = json.load(f)

        self.general_config_dict["simu_name"] = self.simu_name
        self.general_config_dict["eeg_snr"] = "infdb"

        if self.general_config_dict["source_space"]["constrained_orientation"]:
            self.ori = "constrained"
        else:
            self.ori = "unconstrained"
        # Build data folder name from the provided `root_simu` (portable across OSes)
        data_folder_name = os.path.normpath(
            os.path.join(
                self.root_simu, self.ori, self.electrode_montage, self.source_space, "simu"
            )
        )
        

        (
            self.ids,
            self.eeg_dict,
            self.src_dict,
            self.match_dict,
        ) = utl_data.get_matching_info(data_folder_name, self.general_config_dict, self.root_simu)
        
        #length of ids
        # print("Ids" , len(self.ids))

        # n_times = self.general_config_dict["rec_info"]["n_times"]
        self.max_eeg = torch.zeros((self.to_load, 1))
        self.max_src = torch.zeros((self.to_load, 1))

        self.snr_db = snr_db
        self.noise_type = noise_type

        # Reading thousands of small per-sample metadata JSON files one at a time is slow
        # on a mounted/networked filesystem (e.g. a Kaggle dataset mount) -- each file open
        # pays round-trip latency with nothing overlapping it. Load them concurrently with a
        # thread pool (I/O-bound, so threads -- not processes -- are the right tool) and show
        # a progress bar so this doesn't look like a hang on large datasets.
        self.md_dict = {}

        def _load_one(i):
            with open(self.match_dict[i], "r") as f:
                return i, json.load(f)

        try:
            from tqdm.auto import tqdm
        except ModuleNotFoundError:
            tqdm = lambda it, **kw: it  # noqa: E731

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4) * 4)) as ex:
            for i, data in tqdm(
                ex.map(_load_one, self.ids),
                total=len(self.ids),
                desc="Loading sample metadata",
            ):
                self.md_dict[i] = data

    def __len__(self):
        return min(self.to_load, len(self.ids))

    def __getitem__(self, index):
        ## convert index to id ##
        id = [self.ids[index]]
        ## load data ##
        eeg_data = utl_data.load_eeg_data(
            self.eeg_dict, self.general_config_dict, id, as_tensor=True
        )
        src_data = utl_data.load_src_extended_data(
            js_src=self.src_dict,
            js_md=self.match_dict,
            general_config_dict=self.general_config_dict,
            ids=id,
            as_tensor=True,
        )

        ## add noise ##
        eeg_data, src_data = eeg_data.squeeze(), src_data.squeeze()
        mins_clean = eeg_data.min()
        maxs_clean = eeg_data.max()
        if self.snr_db == "random":
            snr_db = np.random.randint(-5, 10, 1)
        else:
            self.snr_db = int(self.snr_db)
            snr_db = self.snr_db
        eeg_data = utl_data.add_noise_snr(
            snr_db=snr_db, signal=eeg_data, noise_type=self.noise_type
        )
        ### rescale
        eeg_data_bis = utl_data.tensor_range_scaling(
            eeg_data, inf=mins_clean, sup=maxs_clean
        )
        ### normalize ##
        # max-max normalisation
        if self.norm == "max-max":
            self.max_eeg[index] = eeg_data_bis.abs().max()
            self.max_src[index] = src_data.abs().max()
            eeg_data = eeg_data_bis / eeg_data_bis.abs().max()
            src_data = src_data / src_data.abs().max()
        # or linear normalisation
        else:
            self.max_eeg[index] = eeg_data_bis.abs().max()
            self.max_src[index] = eeg_data_bis.abs().max()  # usefull for unscaling
            eeg_data = eeg_data_bis / eeg_data_bis.abs().max()
            src_data = src_data / eeg_data_bis.abs().max()
            
            
        # save as mat file with index as the file name 
        # need only one mat with bot eeg anfd source data
        #savemat(f"eeg_and_src_data_{index}.mat", {"eeg_data": eeg_data, "src_data": src_data})

        return eeg_data, src_data


################## modified
import os


class ModSpikeEEGBuild(Dataset):

    """Dataset, generate input/output on the run

    Attributes
    ----------
    data_root : str
        Dataset file location
    fwd : np.array
        Size is num_electrode * num_region
    dataset_meta : dict
        Information needed to generate data
        selected_region: spatial model for the sources; num_examples * num_sources * max_size
                         num_examples: num_examples in this dataset
                         num_sources: num_sources in one example
                         max_size: cortical regions in one source patch; first value is the center region id; variable length, padded to max_size
                            (set to 70, an arbitrary number)
        nmm_idx:         num_examples * num_sources: index of the TVB data to use as the source
        scale_ratio:     scale the waveform maginitude in source region; num_examples * num_sources * num_scale_ratio (num_snr_level)
        mag_change:      magnitude changes inside a source patch; num_examples * num_sources * max_size
                         weight decay inside a patch; equals to 1 in the center region; variable length; padded to max_size
        sensor_snr:      the Gaussian noise added to the sensor space; num_examples * 1;

    dataset_len : int
        size of the dataset, can be set as a small value during debugging
    """

    def __init__(
        self,
        spike_data_path,
        metadata_file,
        fwd,
        spos,
        n_times=500,
        transform=None,
        args_params=None,
        norm="linear",
    ):

        # args_params: optional parameters; can be dataset_len
        self.norm = norm
        self.metadata_file_path = metadata_file
        self.spike_data_path = spike_data_path
        self.fwd = fwd
        self.transform = transform
        self.n_times = n_times
        self.spos = spos  # source positions, used to compute weight decay

        self.dataset_meta = loadmat(self.metadata_file_path)
        if "dataset_len" in args_params:
            self.dataset_len = args_params["dataset_len"]
        else:  # use the whole dataset
            self.dataset_len = self.dataset_meta["selected_region"].shape[0]
        if "num_scale_ratio" in args_params:
            self.num_scale_ratio = args_params["num_scale_ratio"]
        else:
            self.num_scale_ratio = self.dataset_meta["scale_ratio"].shape[2]

        ### QUICK FIX OF PROBLEMATIC DATASET
        # self.shitty_results = [
        #    387, 910, 7, 419, 936,
        #    938, 417, 325, 411, 921,
        #    356, 923, 915, 949, 917,
        #    418, 940, 920, 922, 415,
        #    993 ]
        self.shitty_results = []
        self.good_regions = np.setdiff1d(np.arange(0, 994, 1), self.shitty_results)

        self.max_eeg = torch.zeros((self.dataset_len), 1)
        self.max_src = torch.zeros((self.dataset_len), 1)

        ### select nmm index
        for k in range(self.dataset_meta["random_samples"].shape[0]):
            raw_lb = self.dataset_meta["selected_region"][k].astype(
                int
            )  # labels with padding
            #lb = raw_lb[np.logical_not(ispadding(raw_lb))]  # labels without padding

            for kk in range(raw_lb.shape[0]):  # iterate through number of sources
                curr_lb = raw_lb[kk, np.logical_not(ispadding(raw_lb[kk]))]
                a_center_kk = curr_lb[0]

                n_nmm_clips = len(os.listdir(f"{self.spike_data_path}/a{a_center_kk}"))
                if n_nmm_clips == 1:
                    self.dataset_meta["random_samples"][k][kk] = 1
                else:
                    self.dataset_meta["random_samples"][k][kk] = np.random.randint(
                        1, n_nmm_clips
                    )

    def __getitem__(self, index):

        raw_lb = self.dataset_meta["selected_region"][index].astype(
            int
        )  # labels with padding
        lb = raw_lb[np.logical_not(ispadding(raw_lb))]  # labels without padding

        raw_nmm     = np.zeros((self.n_times, self.fwd.shape[1]))
        noise_nmm   = np.zeros((self.n_times, self.fwd.shape[1])) #

        # for kk in range(raw_lb.shape[0]):                                           # iterate through number of sources
        #    curr_lb = raw_lb[kk, np.logical_not(ispadding(raw_lb[kk]))]
        #    a_center_kk = curr_lb[0]
        #
        #    if a_center_kk in self.good_regions :
        #
        #        n_nmm_clips = len(
        #            os.listdir(f"{self.spike_data_path}/a{a_center_kk}")
        #        )
        #        if n_nmm_clips == 1:
        #            current_nmm = loadmat(
        #                f"{self.spike_data_path}/a{a_center_kk}/nmm_1.mat"
        #            )['data'] # load a random clip
        #        else :
        #            current_nmm = loadmat(
        #                f"{self.spike_data_path}/a{a_center_kk}/nmm_{np.random.randint(1,n_nmm_clips)}.mat"
        #            )['data']
        #    else :
        #        replacement_a_center = np.random.choice(self.good_regions, 1).item()
        #        n_nmm_clips = len(
        #            os.listdir(f"{self.spike_data_path}/a{replacement_a_center}")
        #        )
        #        if n_nmm_clips == 1:
        #            current_nmm = loadmat(
        #                f"{self.spike_data_path}/a{replacement_a_center}/nmm_1.mat"
        #            )['data'] # load a random clip
        #        else :
        #            current_nmm = loadmat(
        #                f"{self.spike_data_path}/a{replacement_a_center}/nmm_{np.random.randint(1,n_nmm_clips)}.mat"
        #            )['data']
        #        sig = current_nmm[:,replacement_a_center]
        #        current_nmm[ : , replacement_a_center ] = current_nmm[:, a_center_kk]
        #        current_nmm[:, a_center_kk] = np.copy(sig)
        #        del sig
        #
        for kk in range(raw_lb.shape[0]):  # iterate through number of sources
            curr_lb = raw_lb[kk, np.logical_not(ispadding(raw_lb[kk]))]
            a_center_kk = curr_lb[0]

            current_nmm = loadmat(
                f"{self.spike_data_path}/a{a_center_kk}/nmm_{self.dataset_meta['random_samples'][index][kk]}.mat"
            )["data"]

            ssig = current_nmm[:, [curr_lb[0]]]  # waveform in the center region
            # set source space SNR
            ssig = (
                ssig
                / np.max(ssig)
                * self.dataset_meta["scale_ratio"][index][kk][
                    random.randint(0, self.num_scale_ratio - 1)
                ]
            )
            current_nmm[:, curr_lb] = ssig.reshape(-1, 1)
            # set weight decay inside one source patch
            #weight_decay = self.dataset_meta["mag_change"][index][kk]
            #weight_decay = weight_decay[np.logical_not(ispadding(weight_decay))]

            ### ------- change in the computation of weight decay for amplitude decay ---------###
            d_in_patch = np.sqrt(
                np.sum((self.spos[curr_lb[0], :] - self.spos[curr_lb, :]) ** 2, 1)
            )
            sig = (np.max(d_in_patch)) / np.sqrt(2 * np.log(2))
            weight_decay = np.exp(-0.5 * (d_in_patch / sig) ** 2)
            ###
            #current_nmm[:, curr_lb] = ssig.reshape(-1, 1) * weight_decay

            #raw_nmm = raw_nmm + current_nmm
            noise_nmm = noise_nmm + current_nmm
            raw_nmm[:,curr_lb] = ssig.reshape(-1, 1) * weight_decay

        noise_nmm /= len(raw_lb)
        unnoisy_sources = np.setdiff1d(np.arange(994), lb)
        raw_nmm[:,unnoisy_sources] = noise_nmm[:, unnoisy_sources]

        eeg = np.matmul(
            self.fwd, raw_nmm.transpose()
        )  # project data to sensor space; num_electrode * num_time
        csnr = self.dataset_meta["current_snr"][index]
        noisy_eeg = add_white_noise(eeg, csnr).transpose()

        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=0, keepdims=True)  # time
        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=1, keepdims=True)  # channel

        # get the training output
        empty_nmm = np.zeros_like(raw_nmm)
        empty_nmm[:, lb] = raw_nmm[:, lb]

        ## normalize data
        self.max_eeg[index] = np.max(np.abs(noisy_eeg))
        noisy_eeg = noisy_eeg / np.max(np.abs(noisy_eeg))
        if self.norm == "max-max":
            self.max_src[index] = np.max(np.abs(empty_nmm))
        #    empty_nmm = empty_nmm / np.max(np.abs(empty_nmm))
        else:
            self.max_src[index] = self.max_eeg[index]
        #    empty_nmm = empty_nmm / np.max(np.abs(noisy_eeg))

        # Each data sample
        sample = {
            "data": noisy_eeg.astype("float32"),
            "nmm": empty_nmm.astype("float32"),
            "label": raw_lb,
            "snr": csnr,
        }
        if self.transform:
            sample = self.transform(sample)

        # savemat('{}/data{}.mat'.format(self.file_path[0][:-4],index),{'data':noisy_eeg,'label':raw_lb,'nmm':empty_nmm[:,lb]})
        return torch.from_numpy(sample["data"].transpose()), torch.from_numpy(
            sample["nmm"].transpose()
        )

    def __len__(self):
        return self.dataset_len


class SpikeEEGLoad(Dataset):

    """Dataset, load pregenerated input/output pair

    Attributes
    ----------
    data_root : str
        Dataset file location
    fwd : np.array
        Size is num_electrode * num_region
    dataset_len : int
        size of the dataset, can be set as a small value during debugging
    """

    def __init__(self, data_root, fwd, transform=None, args_params=None):

        # args_params: optional parameters; can be dataset_len

        self.file_path = data_root
        self.fwd = fwd
        self.transform = transform
        if "dataset_len" in args_params:
            self.dataset_len = args_params["dataset_len"]
        else:  # use the whole dataset
            self.dataset_len = len(dir("{}/data*.mat"))

    def __getitem__(self, index):

        # load data saved as separate files using loadmat
        raw_data = loadmat("{}/data{}".format(self.file_path, index))
        sample = {
            "data": raw_data["data"].astype("float32"),
            "nmm": raw_data["nmm"].astype("float32"),
            "label": raw_data["label"],
            "snr": raw_data["csnr"],
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.dataset_len


#####################################################################
## did not check thath
'''
class SpikeEEGBuildEval(Dataset):

    """Dataset, generate test data under different conditions to evaluate the model under different conditions

    Attributes
    ----------
    data_root : str
        Dataset file location
    fwd : np.array
        Size is num_electrode * num_region
    data : np.array
        TVB output data
    dataset_meta : dict
        Information needed to generate data
        selected_region: spatial model for the sources; num_examples * num_sources * max_size
                         num_examples: num_examples in this dataset
                         num_sources: num_sources in one example
                         max_size: cortical regions in one source patch; first value is the center region id; variable length, padded to max_size
                            (set to 70, an arbitrary number)
        nmm_idx:         num_examples * num_sources: index of the TVB data to use as the source
        scale_ratio:     scale the waveform maginitude in source region; num_examples * num_sources * num_scale_ratio (num_snr_level)
        mag_change:      magnitude changes inside a source patch; num_examples * num_sources * max_size
                         weight decay inside a patch; equals to 1 in the center region; variable length; padded to max_size
        sensor_snr:      the Gaussian noise added to the sensor space; num_examples * 1;

    dataset_len : int
        size of the dataset, can be set as a small value during debugging

    eval_params : dict
        New attributes compare to SpikeEEGBuild, depending on the test running, keys can be
        lfreq :         int; high pass cut-off frequency; filter EEG data to perform narrow-band analysis
        hfreq :         int; low pass cut-off frequency; filter EEG data to perform narrow-band analysis
        snr_rsn_ratio:  float; [0, 1]; ratio between real noise and gaussian noise


    """

    def __init__(self, data_root, fwd, transform=None, args_params=None):

        # args_params: optional parameters; can be dataset_len, num_scale_ratio

        self.file_path = data_root
        self.fwd = fwd
        self.transform = transform

        self.data = []
        self.dataset_meta = loadmat(self.file_path)
        self.eval_params = dict()

        # check args_params:
        if "dataset_len" in args_params:
            self.dataset_len = args_params["dataset_len"]
        else:  # use the whole dataset
            self.dataset_len = self.dataset_meta["selected_region"].shape[0]
        if "num_scale_ratio" in args_params:
            self.num_scale_ratio = args_params["num_scale_ratio"]
        else:
            self.num_scale_ratio = self.dataset_meta["scale_ratio"].shape[2]

        if (
            "snr_rsn_ratio" in args_params and args_params["snr_rsn_ratio"]
        ):  # Need to add realistic noise
            self.eval_params["rsn"] = loadmat("anatomy/realistic_noise.mat")
            self.eval_params["snr_rsn_ratio"] = args_params["snr_rsn_ratio"]
        if "lfreq" in args_params and args_params["lfreq"] > 0:
            if "hfreq" in args_params and args_params["hfreq"] > 0:
                self.eval_params["lfreq"] = args_params["lfreq"]
                self.eval_params["hfreq"] = args_params["hfreq"]
            else:
                print(
                    "WARNING: NEED TO ASSIGN BOTH LOW-PASS AND HIGH-PASS CUT-OFF FREQ, IGNORE FILTERING"
                )

    def __getitem__(self, index):

        if not self.data:
            self.data = h5py.File("{}_nmm.h5".format(self.file_path[:-12]), "r")["data"]

        raw_lb = self.dataset_meta["selected_region"][index].astype(
            int
        )  # labels with padding
        lb = raw_lb[np.logical_not(ispadding(raw_lb))]  # labels without padding
        raw_nmm = np.zeros((500, self.fwd.shape[1]))

        for kk in range(raw_lb.shape[0]):  # iterate through number of sources
            curr_lb = raw_lb[kk, np.logical_not(ispadding(raw_lb[kk]))]
            current_nmm = self.data[self.dataset_meta["nmm_idx"][index][kk]]

            ssig = current_nmm[:, [curr_lb[0]]]  # waveform in the center region
            # set source space SNR
            ssig = (
                ssig
                / np.max(ssig)
                * self.dataset_meta["scale_ratio"][index][kk][
                    random.randint(0, self.num_scale_ratio - 1)
                ]
            )
            current_nmm[:, curr_lb] = ssig.reshape(-1, 1)
            # set weight decay inside one source patch
            weight_decay = self.dataset_meta["mag_change"][index][kk]
            weight_decay = weight_decay[np.logical_not(ispadding(weight_decay))]
            current_nmm[:, curr_lb] = ssig.reshape(-1, 1) * weight_decay

            raw_nmm = raw_nmm + current_nmm

        eeg = np.matmul(
            self.fwd, raw_nmm.transpose()
        )  # project data to sensor space; num_electrode * num_time
        csnr = self.dataset_meta["sensor_snr"][index]

        # add noise to sensor space
        if "rsn" in self.eval_params:
            noisy_eeg = add_white_noise(
                eeg,
                csnr,
                {
                    "ratio": self.eval_params["snr_rsn_ratio"],
                    "rndata": self.eval_params["rsn"]["data"],
                    "rnpower": self.eval_params["rsn"]["npower"],
                },
            ).transpose()
        else:
            noisy_eeg = add_white_noise(eeg, csnr).transpose()

        # filter data into narrow band
        if "lfreq" in self.eval_params:
            # To reduce border effects, tile in time then crop back.
            tiled = np.tile(noisy_eeg.transpose(), (1, 5))  # (n_ch, n_time*5)
            fs = 500
            lfreq = self.eval_params["lfreq"]
            hfreq = self.eval_params["hfreq"]
            if HAS_MNE:
                filtered = mne.filter.filter_data(
                    tiled,
                    fs,
                    lfreq,
                    hfreq,
                    verbose=False,
                )
            else:
                # Lightweight fallback without MNE
                from scipy.signal import butter, filtfilt

                nyq = fs / 2.0
                b, a = butter(4, [lfreq / nyq, hfreq / nyq], btype="band")
                filtered = filtfilt(b, a, tiled, axis=1)

            noisy_eeg = filtered.transpose()
            noisy_eeg = noisy_eeg[1000:1500]

        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=0, keepdims=True)  # time
        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=1, keepdims=True)  # channel
        noisy_eeg = noisy_eeg / np.max(np.abs(noisy_eeg))

        # get the training output
        empty_nmm = np.zeros_like(raw_nmm)
        empty_nmm[:, lb] = raw_nmm[:, lb]
        empty_nmm = empty_nmm / np.max(empty_nmm)
        # Each data sample
        sample = {
            "data": noisy_eeg.astype("float32"),
            "nmm": empty_nmm.astype("float32"),
            "label": raw_lb,
            "snr": csnr,
        }
        if self.transform:
            sample = self.transform(sample)

        # savemat('{}/data{}.mat'.format(self.file_path[0][:-4],index),{'data':noisy_eeg,'label':raw_lb,'nmm':empty_nmm[:,lb]})
        return sample

    def __len__(self):
        return self.dataset_len
'''

"""
# old one which load all data at the same time
class EsiDatasetds(Dataset):
    def __init__(
        self,
        config_file,
        simu_name,
        source_space,
        electrode_montage,
        to_load,
        eeg_snr,
        noise_type={"white": 1.0},
    ) -> None:
        super().__init__()
        home = expanduser("~")
        self.simu_name = simu_name
        self.source_space = source_space
        self.electrode_montage = electrode_montage
        self.to_load = to_load
        self.eeg_snr = eeg_snr

        self.config_file = config_file

        with open(config_file, "r") as f:
            self.general_config_dict = json.load(f)

        self.general_config_dict["simu_name"] = self.simu_name
        self.general_config_dict["eeg_snr"] = "infdb"

        if self.general_config_dict["source_space"]["constrained_orientation"]:
            self.ori = "constrained"
        else:
            self.ori = "unconstrained"
        # build data folder name
        data_folder_name = f"{home}/Documents/Data/simulation/{self.ori}/{self.electrode_montage}/{self.source_space}/simu"

        ids, eeg_dict, src_dict, md_dict = utl_data.get_matching_info(
            data_folder_name, self.general_config_dict
        )

        n_times = self.general_config_dict["rec_info"]["n_times"]
        shuffled_indices = torch.randperm(self.to_load)
        self.ds_ids = [ids[i] for i in shuffled_indices]

        self.eeg_data = utl_data.load_eeg_data(
            eeg_dict, self.general_config_dict, self.ds_ids, as_tensor=True
        )
        self.src_data = utl_data.load_src_extended_data(
            js_src=src_dict,
            js_md=md_dict,
            general_config_dict=self.general_config_dict,
            ids=self.ds_ids,
            as_tensor=True,
        )

        # add noise if necessary
        mins_clean = self.eeg_data.view(self.eeg_data.shape[0], -1).min(dim=1)[0]
        maxs_clean = self.eeg_data.view(self.eeg_data.shape[0], -1).max(dim=1)[0]

        if self.eeg_snr < 50:
            snr_range = [-5, 10]
            self.snr_db = np.expand_dims(
                np.random.randint(snr_range[0], snr_range[1], self.eeg_data.shape[0]),
                (1, 2),
            )

            #self.eeg_data = utl_data.add_noise_snr(
            #    snr_db=self.snr_db, signal=self.eeg_data, noise_type=noise_type
            #)
            ## rescale
            #self.eeg_data = utl_data.tensor_range_scaling(
            #    self.eeg_data, inf=mins_clean, sup=maxs_clean
            #)

        ### SCALING
        self.max_eeg = (
            self.eeg_data.view(self.eeg_data.shape[0], -1).abs().max(dim=1)[0]
        )
        self.max_src = (
            self.src_data.view(self.src_data.shape[0], -1).abs().max(dim=1)[0]
        )
        self.eeg_data /= self.max_eeg.view(-1, 1, 1)
        self.src_data /= self.max_src.view(-1, 1, 1)

        self.md_dict = {}
        self.src_dict = {}
        self.eeg_dict = {}
        for i in self.ds_ids:
            with open(md_dict[i], "r") as f:
                self.md_dict[i] = json.load(f)

            # self.md_dict[i] = md_dict[ i ]
            self.src_dict[i] = src_dict[i]
            self.eeg_dict[i] = eeg_dict[i]

    def __getitem__(self, index):
        
        return self.eeg_data[index, :, :], self.src_data[index, :, :]

    def __len__(self):
        return self.eeg_data.shape[0]
"""
