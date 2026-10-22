import sys

import numpy as np
from utils.utl import load_mat
import os

"""
This module originally relied on `mne` for:
- Loading forward solutions from FIF files
- Creating montage/info objects for EEG

For environments where `mne` is not available or not desired, this file now
supports a "no-mne" workflow:
- Leadfield is loaded from a `.mat` file (e.g. `anatomy/leadfield_75_20k.mat`)
- Electrode space is described by basic arrays (names, positions, fs)
"""

class ElectrodeSpace:
    """  
    Get, build and store information about the electrode space

    - n_electrodes      : number of electrodes
    - positions         : positions of the electrodes
    - montage_kind      : name of the electrode montage used
    - electrode_names   : name of the electrodes
    - electrode_montage : electrode montage of mne-python (DigMontage),
                         useful to manipulate eeg data
    - info              : info object from mne-python
    - fs                : sampling frequency
    

    @TODO : add visualisation function to plot electrodes in 2D or 3D
    """

    def __init__(self, folders, general_config_dict):
        """
        - folders: FolderStructure object containing all the name of the folders
        - general_config_dict: dictionnary with information about simulation configuration
        """

        montage_kind = general_config_dict['electrode_space']['electrode_montage']

        # load the ch_source_sampling.mat file which contains basic information and data of the electrode space
        ch_mat_path = f"{folders.model_folder}/ch_{general_config_dict['source_space']['src_sampling']}.mat"
        if os.path.isfile(ch_mat_path):
            electrode_info = load_mat(ch_mat_path)
            self.n_electrodes = electrode_info['nb_channels']
            self.positions = electrode_info['positions']
            self.electrode_names = [k for k in electrode_info['names']]
        else:
            # Fallback (mirrors SourceSpace's anatomy/ fallback below): some setups only
            # ship the `simu/` data (e.g. a dataset uploaded to Kaggle without the
            # `model/` head-model folder). Build minimal electrode info from the
            # repo-level anatomy file instead, matched by montage name.
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            fallback_candidates = []
            if montage_kind == "standard_1020":
                fallback_candidates.append(os.path.join(repo_root, "anatomy", "electrode_75.mat"))

            fallback_path = next((p for p in fallback_candidates if os.path.isfile(p)), None)
            if fallback_path is None:
                raise FileNotFoundError(
                    f"Could not find '{ch_mat_path}', and no repo-level electrode-info "
                    f"fallback is known for montage '{montage_kind}'. "
                    f"Tried: {fallback_candidates}"
                )

            from scipy.io import loadmat as _loadmat

            m = _loadmat(fallback_path, struct_as_record=False, squeeze_me=True)
            struct_key = next(k for k in m if not k.startswith("__"))
            eloc = m[struct_key]
            self.n_electrodes = len(eloc)
            self.positions = np.array([[e.X, e.Y, e.Z] for e in eloc], dtype=float)
            self.electrode_names = [str(e.labels) for e in eloc]

        self.montage_kind = montage_kind

        # In "no-mne" mode we keep only lightweight descriptors.
        # If you need MNE objects, build them in a separate optional module.
        rec_info = general_config_dict.get("rec_info", {}) if isinstance(general_config_dict, dict) else {}
        self.fs = rec_info.get("fs", None)
        self.electrode_montage = None
        self.info = None

    def _attributes(self):
        """
        return the liste of attribute/variables of the object
        """
        return list(self.__dict__.keys())


class SourceSpace:
    """  
    - src_sampling  : name of the source subsampling used to subsample the source space ('oct3', 'ico3'...)
    - n_sources     : number of sources
    - constrained   : True if constrained orientation, False if unconstrained
    - positions     : source positions
    - orientations  : source orientations (values are filled during HeadModel initialization)

    @TODO: add visualisation of source positions
    """
    def __init__(self, folders, general_config_dict, surface=True, volume=False):
        self.src_sampling   = general_config_dict['source_space']['src_sampling']
        #self.n_sources      = general_config_dict['source_space']['n_sources']
        self.constrained    = general_config_dict['source_space']['constrained_orientation']

        # Source geometry location:
        # - for typical source spaces: stored under the head-model folder (folders.model_folder)
        # - for the 994-region parcellation: stored under the repo-level `anatomy/` folder
        if self.src_sampling == "fsav_994":
            # Try to resolve relative to the current run root (folders.root_folder) first.
            # `folders.root_folder` may be either the repo root or `.../simulation/<subject>`.
            candidates = []
            try:
                root = os.path.abspath(folders.root_folder)
                candidates.append(os.path.join(root, "anatomy", "sources_fsav_994.mat"))
                candidates.append(
                    os.path.join(os.path.dirname(root), "anatomy", "sources_fsav_994.mat")
                )
            except Exception:
                pass

            # Fallback: repo-relative to this file location
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            candidates.append(os.path.join(repo_root, "anatomy", "sources_fsav_994.mat"))

            source_mat_path = next((p for p in candidates if os.path.isfile(p)), None)
            if not source_mat_path:
                raise FileNotFoundError(
                    "Could not find `sources_fsav_994.mat`. Tried:\n- "
                    + "\n- ".join(candidates)
                )
            source_info = load_mat(source_mat_path)
        else:
            source_info = load_mat(f"{folders.model_folder}/sources_{self.src_sampling}.mat")

        self.positions = source_info['positions']
        self.n_sources = self.positions.shape[0]
        self.orientations = []  # to complete

        # useless for now
        self.surface = surface
        self.volume = volume

    def _attributes(self):
        """
        return the liste of attribute/variables of the object
        """
        return list(self.__dict__.keys())

class HeadModel:
    """  
    Gather electrode space and source space + forward solution
    - electrode_space   : ElectrodeSpace object
    - source_space      : SourceSpace object
    - subject_name      : default is 'fsaverage', name of the subject used.
    - fwd               : mne python Forward ojbect created during head model generation
    - leadfield         : leadfield matrix

    @TODO : add visualizaion of electrode and sources
    """
    def __init__(self, electrode_space, source_space, folders, subject_name='fsaverage'):
        self.electrode_space    = electrode_space
        self.source_space       = source_space

        self.subject_name       = subject_name
        # "no-mne" forward structure: we only keep the leadfield matrix in a dict
        # compatible with the rest of the codebase (`fwd['sol']['data']`).
        self.fwd = {"sol": {"data": None}}
        self.leadfield = None

        # If available, override with the repository-provided leadfield matrix
        # (keeps `self.fwd` structure but sets the gain matrix).
        #
        # Expected file: <repo_root>/anatomy/leadfield_75_20k.mat
        # Common keys inside the .mat: 'fwd' or 'G'
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        default_lf_mat_path = os.path.join(repo_root, "anatomy", "leadfield_75_20k.mat")

        # Allow overriding via env var, and keep a Windows fallback used elsewhere in the repo.
        lf_candidates = []
        env_path = os.environ.get("STESI_LEADFIELD_MAT", "").strip()
        if env_path:
            lf_candidates.append(env_path)
        lf_candidates.append(default_lf_mat_path)
        if os.name == "nt":
            lf_candidates.append(r"D:\fyp\stESI_pub\anatomy\leadfield_75_20k.mat")

        lf_mat_path = next((p for p in lf_candidates if os.path.isfile(p)), None)
        if not lf_mat_path:
            raise FileNotFoundError(
                "No leadfield file found. Tried:\n- "
                + "\n- ".join(lf_candidates)
                + "\nSet STESI_LEADFIELD_MAT to point to your leadfield .mat."
            )

        try:
            from scipy.io import loadmat

            lf_mat = loadmat(lf_mat_path)
            lf = None
            for k in ("fwd", "G", "leadfield"):
                if k in lf_mat:
                    lf = lf_mat[k]
                    break
            if lf is None:
                raise KeyError(
                    f"Could not find leadfield array in {lf_mat_path}. "
                    f"Available keys: {sorted([k for k in lf_mat.keys() if not k.startswith('__')])}"
                )

            self.fwd["sol"]["data"] = lf
            self.leadfield = lf
        except Exception as e:
            raise RuntimeError(
                f"Failed to load leadfield from {lf_mat_path}: {e}"
            ) from e

        # add orientation to source space
        # Not available without MNE forward; keep empty unless provided elsewhere.
        self.source_space.orientations = getattr(self.source_space, "orientations", [])

    def _attributes(self):
        """
        return the liste of attribute/variables of the object
        """
        return list(self.__dict__.keys())