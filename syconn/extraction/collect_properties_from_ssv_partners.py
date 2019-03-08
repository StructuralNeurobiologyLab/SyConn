def collect_properties_from_ssv_partners(wd, obj_version=None, ssd_version=None,
                                         qsub_pe=None, qsub_queue=None,
                                         nb_cpus=None, n_max_co_processes=None):
    """
    Collect axoness, cell types and spiness from synaptic partners and stores
    them in syn_ssv objects. Also maps syn_type_sym_ratio to the synaptic sign
    (-1 for asym., 1 for sym. synapses).

    Parameters
    ----------
    wd : str
    obj_version : str
    ssd_version : int
    qsub_pe : str
    qsub_queue : str
    nb_cpus : int
        CPUS per job
    n_max_co_processes : int
        Number of parallel jobs
    """
    # ssd = SuperSegmentationDataset(working_dir=wd)
    # ssv_ids = ssd.ssv_ids

    sd_syn_ssv = segmentation.SegmentationDataset("syn_ssv", working_dir=wd,
                                                  version=obj_version)
    multi_params = []
    for so_dir_paths in chunkify(sd_syn_ssv.so_dir_paths, 2000):
        multi_params.append([so_dir_paths, wd, obj_version,
                             ssd_version])
    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        _ = sm.start_multiprocess_imap(
            _collect_properties_from_ssv_partners_thread, multi_params,
            nb_cpus=n_max_co_processes)
    elif qu.batchjob_enabled():
        _ = qu.QSUB_script(
            multi_params, "collect_properties_from_ssv_partners", pe=qsub_pe,
            queue=qsub_queue, script_folder=None,
            n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")


def _collect_properties_from_ssv_partners_thread(args):
    """
    Helper function of 'collect_properties_from_ssv_partners'.

    Parameters
    ----------
    args : Tuple
        see 'collect_properties_from_ssv_partners'
    """
    so_dir_paths, wd, obj_version, ssd_version = args

    ssd = super_segmentation.SuperSegmentationDataset(working_dir=wd,
                                                      version=ssd_version)
    sd_syn_ssv = segmentation.SegmentationDataset(obj_type="syn_ssv",
                                                  working_dir=wd,
                                                  version=obj_version)
    # syn_neuronpartners = sd_syn_ssv.load_cached_data("neuron_partners")
    for so_dir_path in so_dir_paths:
        this_attr_dc = AttributeDict(so_dir_path + "/attr_dict.pkl",
                                     read_only=False)
        # ssv_synids = sd_syn_ssv.ids[sd_synssv.load_cached_data("neuron_partners")[:, 0] == ssv_id]
        for synssv_id in this_attr_dc.keys():
            synssv_o = sd_syn_ssv.get_segmentation_object(synssv_id)
            synssv_o.load_attr_dict()

            axoness = []
            spiness = []
            celltypes = []
            for ssv_partner_id in synssv_o.attr_dict["neuron_partners"]:

                ssv_o = ssd.get_super_segmentation_object(ssv_partner_id)
                ssv_o.load_attr_dict()
                # add pred_type key to global_params?
                curr_ax = ssv_o.axoness_for_coords([synssv_o.rep_coord],
                                                   pred_type='axoness_avg10000')
                axoness.append(curr_ax[0])
                # TODO: maybe use more than only a single rep_coord
                curr_sp = ssv_o.semseg_for_coords([synssv_o.rep_coord],
                                                  'spiness')
                spiness.append(curr_sp)
                celltypes.append(ssv_o.attr_dict['celltype_cnn'])
            sym_asym_ratio = synssv_o.attr_dict['syn_type_sym_ratio']
            syn_sign = -1 if sym_asym_ratio > global_params.sym_thresh else 1
            synssv_o.attr_dict.update({'partner_axoness': axoness, 'partner_spiness': spiness,
                                       'partner_celltypes': celltypes, 'syn_sign': syn_sign})
            this_attr_dc[synssv_id] = synssv_o.attr_dict
        this_attr_dc.push()

