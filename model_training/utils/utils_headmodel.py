import matplotlib
import mne
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

###################### subject ######################

def create_head_model(
    subject="fsaverage",
    montage_kind="easycap-M10",
    sampling="oct5",
    constrained=True,
    volume=False, 
    conductivity = (0.3, 0.006, 0.3) ,
    fs=1000):

    fs_dir       = mne.datasets.fetch_fsaverage(verbose = 0)
    subjects_dir = os.path.dirname(fs_dir)

    trans = os.path.join(fs_dir, 'bem', 'fsaverage-trans.fif')
    ########## BEM model ##################
    model   = mne.make_bem_model(subject=subject, ico=4,
                conductivity=conductivity,
                subjects_dir=subjects_dir)
    bem     = mne.make_bem_solution(model)

    ############# source space #############
    if not volume :
        src      = mne.setup_source_space( 
            subject, 
            spacing = sampling,
            surface = 'white', 
            subjects_dir = subjects_dir, 
            add_dist = False, 
            n_jobs = -1, 
            verbose = 0)
    else : 
        mri   = os.path.join(fs_dir, 'mri', 'brain.mgz')
        src      = mne.setup_volume_source_space( 
            subject, 
            pos = 10.0, 
            mri = mri,
            bem =  bem,
            subjects_dir = subjects_dir,
            verbose = 0)

    ############ electrode space ############
    montage = mne.channels.make_standard_montage(montage_kind)

    if montage_kind == 'standard_1020': 
        # remove duplicate electrodes (modified combinatorial nomenclature)
        exclude_mdn             = ['T3', 'T4', 'T5', 'T6']
        ids_duplicate = []
        for e in exclude_mdn:
            ids_duplicate.append( np.where( [ch==e for ch in montage.ch_names] )[0][0] )
        ch_names = list( np.delete(montage.ch_names, ids_duplicate) )
        fs      = 1000 
        info    = mne.create_info(
                ch_names, 
                fs, 
                ch_types=['eeg']*len(ch_names), verbose=0)
        info.set_montage(montage_kind)
    elif montage_kind == 'easycap-M10': 
        fs      = 1000 
        info    = mne.create_info(
                montage.ch_names, 
                fs, 
                ch_types=['eeg']*len(montage.ch_names), verbose=0)
        info.set_montage(montage_kind)
        sys.exit("Careful with easycap M10 which is not registered to fsaverage model.")
    else: 
        sys.exit("Unknown montage")

    ####### Compute the forward problem solution -> i.e get the leadfield matrix.
    fwd = mne.make_forward_solution(
        info,
        trans = trans, 
        src = src, 
        bem = bem, eeg=True, mindist=5.0, n_jobs = -1, verbose = 0)

    if not volume:
        # if volume source space the conversion to a constrained orientation is not possible.
        fwd = mne.convert_forward_solution(
            fwd, 
            surf_ori= constrained, 
            force_fixed= constrained,
            use_cps=True, verbose=0)
    
    return trans, bem, src, info, fwd

def hm_mne_to_sereega( info, fwd, verbose=True ):
    ### Channels
    # /!\ Channels in head coordinates as the sources
    channel_names     = info.ch_names
    nb_channels       = len(channel_names) #fwd['nchan']

    #digL = info['dig']
    #chPosD = np.zeros( [nb_channels,3] )
    #for i in range(3,nb_channels+3):
    #    chPosD[i-3,:] = digL[i]['r']
    #channel_positions = chPosD 
    channel_positions = np.array( [info.get_montage().get_positions()['ch_pos'][chn] for chn in info['ch_names']]  )

    channels_info = {'nb_channels' : nb_channels,
        'positions' : channel_positions, 
        'names' : channel_names }
    chanlocs_info = {
        "ch_types" :  ["eeg" for i in range(nb_channels) ], 
        "labels" : channel_names, 
        "sph_radius" : np.zeros([nb_channels, 1]), 
        "sph_theta" : np.zeros([nb_channels,1]),
        "sph_phi" : np.zeros([nb_channels, 1]), 
        "theta" : np.zeros([nb_channels, 1]), 
        "radius" : np.zeros([nb_channels, 1]), 
        "X" : channel_positions[:,0], 
        "Y" : channel_positions[:,1], 
        "Z" : channel_positions[:,2], 
        "sph_theta_besa" : np.zeros([nb_channels, 1]),
        "sph_phi_besa" : np.zeros([nb_channels, 1]), 
    }


    ### Sources
    n_sources           = fwd['nsource']
    source_position     = fwd['source_rr'] #*10**3
    source_orientations = fwd['source_nn']
    srcM                = fwd['src'] # SourceSpace object of the model
    if fwd['source_ori']:
        constrained = True
    else: 
        constrained = False
    sources_info = {
        'nb_sources' : n_sources, 
        'positions' : source_position, 
        'orientations' : source_orientations, 
        'constrained' : constrained, 
        'sourceSpace' : srcM, 
    }

    ### Leadfield
    lf              = fwd['sol']['data']
    leadfield = {
        'G': lf
    }

    if verbose: 
        print('nb eeg channels : ', nb_channels)
        print('nb sources : ', n_sources)
        print('Leadfield dimension : ', lf.shape)

    return  channels_info, chanlocs_info, sources_info, leadfield


########### sphere model #########################

def utl_gen_sphere_model( montage_kind = 'easycap-M10', fs = 1000, center = 'auto', rad = 0.09,\
    rel_radii = (0.90, 0.92, 0.97, 1.0), conductivities = (0.33, 1.0, 0.004, 0.33), \
    grid_space = 10.0, min_space = 5.0, constrained = True, verbose = False):

    
    montage = mne.channels.make_standard_montage(montage_kind)
    info = mne.create_info(
        montage.ch_names, 
        fs, 
        ch_types=['eeg']*len(montage.ch_names), verbose=0)
    info.set_montage(montage_kind)

    ## create the sphere model
    sphere = mne.make_sphere_model( r0=center, head_radius = rad, info = info, \
        relative_radii=rel_radii, sigmas = conductivities , verbose = False)

    ## Source space setting (in the sphere created (i.e subsampling) )
    src = mne.setup_volume_source_space( pos = grid_space, sphere=sphere, mindist = min_space, sphere_units = 'm', verbose = False )

    ## Forward model computation with the given configuration 
    fwd = mne.make_forward_solution(info, trans = None, src = src, bem = sphere, \
        meg=False, eeg = True, verbose = False)

    ## constrained sources?
    if constrained : 
        fwd = mne.convert_forward_solution(
            fwd, 
            surf_ori= constrained, 
            force_fixed= constrained,
            use_cps=True, verbose=0)

    # Sensors 
    channel_names     = montage.ch_names
    nb_channels       = fwd['nchan']

    # /!\ Channels in head coordinates as the sources
    digL   = info['dig']
    chPosD = np.zeros( [nb_channels,3] )
    for i in range(3,nb_channels):
        chPosD[i,:] = digL[i]['r'] 

    # Leadfield
    lf              = fwd['sol']['data']

    # Sources
    nb_sources          = fwd['nsource']
    source_position     = fwd['source_rr']
    source_orientations = fwd['source_nn']

    channels_dict = { 
        "positions" : chPosD, 
        "names" : channel_names, 
        'nb_channels' : nb_channels
        }

    chanlocs_dict = {
        "ch_types" :  ["eeg" for i in range(nb_channels) ], 
        "labels" : channel_names, 
        "sph_radius" : np.zeros([nb_channels, 1]), 
        "sph_theta" : np.zeros([nb_channels,1]),
        "sph_phi" : np.zeros([nb_channels, 1]), 
        "theta" : np.zeros([nb_channels, 1]), 
        "radius" : np.zeros([nb_channels, 1]), 
        "X" : chPosD[:,0],
        "Y" : chPosD[:,1], 
        "Z" : chPosD[:,2], 
        "sph_theta_besa" : np.zeros([nb_channels, 1]),
        "sph_phi_besa" : np.zeros([nb_channels, 1]), 
        }

    LF_dict = {
        "G" : lf
        }
 
    sources_dict = { 
        "positions" : source_position, 
        "orientations" : source_orientations,
        "nb_sources" : nb_sources
        }
    
    return channels_dict, chanlocs_dict, sources_dict, LF_dict, fwd 


################## DEEPSIF HEADMODEL  ####################"

def create_deepsif_based_headmodel( region_mapping, conductivity=(0.33, 0.004125, 0.33), fs=500, montage_kind='standard_1020' ):
    from mne.datasets import fetch_fsaverage
    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = os.path.dirname(fs_dir)
    subject = "fsaverage"

    volume = False 
    constrained = True 

    n_regs = np.unique(region_mapping).shape[0] # number of regions

    src_sampling = "ico5" # for fsaverage5 with 20k vertices
    src = mne.setup_source_space( 
        subject, 
        spacing = src_sampling,
        surface = 'pial', 
        subjects_dir = subjects_dir, 
        add_dist = False, 
        n_jobs = -1, 
        verbose = 0)
    
    model   = mne.make_bem_model(subject=subject, ico=4,
                conductivity=conductivity,
                subjects_dir=subjects_dir)
    bem     = mne.make_bem_solution(model)

    montage = mne.channels.make_standard_montage("standard_1020")
    if montage_kind == 'standard_1020': 
        # remove duplicate electrodes (modified combinatorial nomenclature)
        exclude_mdn             = ['T3', 'T4', 'T5', 'T6']
        ids_duplicate = []
        for e in exclude_mdn:
            ids_duplicate.append( np.where( [ch==e for ch in montage.ch_names] )[0][0] )
        ch_names = list( np.delete(montage.ch_names, ids_duplicate) )
        fs      = 1000 
        info    = mne.create_info(
                ch_names, 
                fs, 
                ch_types=['eeg']*len(ch_names), verbose=0)
        info.set_montage(montage_kind)

    trans = os.path.join(fs_dir, 'bem', 'fsaverage-trans.fif') # real trans matrix

    #trans = mne.Transform( fro="mri", to="head", trans=np.eye(4)) # identity to keep source positions which are close to deepSIF's

    fwd = mne.make_forward_solution(
        info,
        trans = trans, 
        src = src, 
        bem = bem, eeg=True, n_jobs = -1, verbose = 0)

    # constrained orientation
    fwd = mne.convert_forward_solution(
        fwd, 
        surf_ori= True, 
        force_fixed= True,
        use_cps=True, verbose=0)
    
    #### REGION MAPPING
    n_sources = src[0]['nuse'] + src[1]['nuse']
    spos= np.zeros((n_sources,3))
    spos[:n_sources//2, :] = src[0]['rr'][ src[0]['vertno'] ] #left hem
    spos[n_sources//2:, :] = src[1]['rr'][ src[1]['vertno'] ] # right hem

    oris = np.zeros((n_sources,3)) #orientations
    oris[:n_sources//2, :] = src[0]['nn'][ src[0]['vertno'] ]  #left hem
    oris[n_sources//2:, :] = src[1]['nn'][ src[1]['vertno'] ]  # right hem

    #labels = [] # list of Label corresponding to each region
    #cdms = {} # store center of mass of each region
    # fake colors for label creation
    reds = np.random.rand(n_regs)
    blues = np.random.rand(n_regs)
    greens = np.random.rand(n_regs)

    pos_cdms = {'rr' : np.zeros( (n_regs, 3) ) , 'nn': np.zeros((n_regs, 3))}  # for creation of discrete source space
    for r in np.unique(region_mapping) : 
        verts = np.where( region_mapping==r )[0]
        if verts[0] > n_sources//2 : 
            hemi = 'rh'
        else :
            hemi = 'lh'
        color = ( reds[r], greens[r], blues[r], 1. )
        lab = mne.Label(vertices=verts, hemi=hemi, name = f"reg_{r}", color = color)
        v_cdm = lab.center_of_mass(subject="fsaverage", restrict_vertices=True)
        pos_cdms['rr'][r,:] = spos[v_cdm,:]
        pos_cdms['nn'][r,:] = oris[v_cdm,:] # !! normal vector to the region = not very correct.

        #labels.append(lab)
    src_region = mne.setup_volume_source_space(
        subject=subject, 
        pos = pos_cdms, 
        bem = bem,
    )
    fwd_region = mne.make_forward_solution(
        info,
        trans = trans, 
        src = src_region, 
        bem = bem, eeg=True, n_jobs = -1, verbose = 0)

    fwd_region = mne.convert_forward_solution(
        fwd_region, 
        surf_ori= True, 
        force_fixed= True,
        use_cps=True, verbose=0)

    n_electrodes = len(ch_names)
    summed_leadfield = np.zeros((n_electrodes, n_regs))
    for r in range(n_regs) : 
        summed_leadfield[:,r] = fwd['sol']['data'][:, np.where( region_mapping==r )[0] ].sum(axis=1)


    return {'trans' : trans, 'bem': bem, 'src_vertices':src, 'info':info, 
            'fwd_vertices':fwd, 'src_regions':src_region, 
            'fwd_regions':fwd_region, 'summed_leadfield':summed_leadfield}
    
