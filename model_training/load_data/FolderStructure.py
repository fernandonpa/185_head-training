class FolderStructure:
    """
    Build and store the different folders 
    """

    def __init__(self, root_folder: str, general_config_dict: dict) -> None:

        constrained         = general_config_dict['source_space']['constrained_orientation']
        electrode_montage   = general_config_dict['electrode_space']['electrode_montage']
        source_sampling     = general_config_dict['source_space']['src_sampling']
        eeg_snr             = general_config_dict['eeg_snr']
        simu_name           = general_config_dict['simu_name']

        self.root_folder    = root_folder

        # 1. data and model folders
        if constrained:
            ori = "constrained"
        else:
            ori = "unconstrained"

        self.data_folder    = f"{self.root_folder}/{ori}/{electrode_montage}/{source_sampling}/simu"
        self.model_folder   = f"{self.root_folder}/{ori}/{electrode_montage}/{source_sampling}/model"

        # 2. simu, eeg, source folders.
        self.simu_folder    = f"{self.data_folder}/{simu_name}"
        self.eeg_folder     = f"{self.simu_folder}/eeg/{eeg_snr}"
        self.source_folder  = f"{self.simu_folder}/sources"

        self.active_source_folder   = f"{self.source_folder}/Jact"
        self.noise_source_folder    = f"{self.source_folder}/Jnoise"
