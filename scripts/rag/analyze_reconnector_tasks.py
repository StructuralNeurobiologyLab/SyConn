# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import glob
from knossos_utils import skeleton
from matplotlib import pyplot as plt
from collections import defaultdict

from syconn.analysis import reconnect_proofreading as repro


def analyze_all_j0126_reconnector_tasks():
    path_to_final_tasks = 'D:/j0126_analysis/reconnect_tasks_batch_1_2_done/j0126_reconnect_tc/'
    path_to_skeletons = 'D:/j0126_analysis/reconnect_tasks_batch_1_2_done/skeletons/'
    kzips = glob.glob(path_to_final_tasks + '*.k.zip')

    parsing_errors = []
    task_dicts = []
    for kzip in kzips:
        try:
            task_dicts.append(repro.analyze_j0126_reconnector_task(kzip))
        except:
            parsing_errors.append(kzip)
            print('Error parsing task {0}'.format(kzip))

    time_per_annotator = defaultdict(list)
    [time_per_annotator[t['annotator'][0]].append(t['mean_task_time']) for t in task_dicts]
    all_task_times = [t['mean_task_time'] for t in task_dicts]
    #for key, val in time_per_annotator.items():
    #    print('Annotator: {0}, mean time {1}'.format(key, np.mean(val)))

    #print('All annotators mean: {0} from {1} reconnects'.format(np.mean(all_task_times), len(all_task_times)*100))

    print('Num parsing errors: {0}, total files: {1}'.format(len(parsing_errors), len(kzips)))

    all_recon_tasks = dict()
    all_recon_tasks['time [s]'] = flatten([t['times'] for t in task_dicts])
    all_recon_tasks['time skel [s]'] = flatten([t['skel_times'] for t in task_dicts])
    all_recon_tasks['length [um]'] = flatten([t['skel_lengths'] for t in task_dicts])
    all_recon_tasks['annotator'] = flatten([t['annotator'] for t in task_dicts])
    all_recon_tasks['k_annos'] = flatten([t['k_annos'] for t in task_dicts])
    all_recon_tasks['src_coords'] = [item for sublist in [t['src_coords'] for t in task_dicts] for item in sublist]
    all_recon_tasks['src_ids'] = flatten([t['src_ids'] for t in task_dicts])

    #all_recon_tasks['pause']
    #all_recon_tasks['kzip']

    skel_length_no_zero = [l for l in all_recon_tasks['length [um]'] if l > 0.1]
    print('Mean skel len no zero: {0}'.format(np.mean(skel_length_no_zero)))

    print('Len time {0}, length {1},'
          'annotator {2}'.format(len(all_recon_tasks['time [s]']),
                                 len(all_recon_tasks['length [um]']),
                                 len(all_recon_tasks['annotator'])))
    no_skel = [t for t in all_recon_tasks['time skel [s]'] if t < 0.1]
    print('Fraction without skeletons: {0}'.format(len(no_skel)
                                                / float(len(all_recon_tasks['time skel [s]']))))
    time_no_skel = []
    for t, skel_t in zip(all_recon_tasks['time [s]'], all_recon_tasks['time skel [s]']):
        if skel_t < 0.1:
            time_no_skel.append(t)

    print('Mean time with no skel: {0}'.format(np.mean(time_no_skel)))
    print('Median time with no skel: {0}'.format(np.median(time_no_skel)))

    skel_paths = []

    #print all_recon_tasks['src_coords']
    for src_id, anno, src_coords in zip(all_recon_tasks['src_ids'],
                                        all_recon_tasks['k_annos'],
                                        all_recon_tasks['src_coords']):

        skel_obj = skeleton.Skeleton()

        #this_anno = skeleton.SkeletonAnnotation()
        #this_anno.scaling = [10.,10., 20.]

        node1 = skeleton.SkeletonNode()
        node1.from_scratch(anno, *src_coords[0])
        node1.setPureComment('source 1')
        anno.addNode(node1)

        node2 = skeleton.SkeletonNode()
        node2.from_scratch(anno, *src_coords[1])
        node2.setPureComment('source 2')
        anno.addNode(node2)

        anno.addEdge(node1, node2)
        #skel_obj.add_annotation(anno)
        skel_obj.add_annotation(anno)
        anno.setComment('')
        outfile = path_to_skeletons + 'reconnect_{0}.nml'.format(src_id)
        #print('Writing {0}'.format(outfile))
        skel_paths.append(outfile)
        skel_obj.toNml(outfile)

    all_recon_tasks['k_annos'] = skel_paths

    # find most difficult ones, indicated by slowest skeletonization speed


    #df_index = [datetime.datetime.fromtimestamp(d).strftime('%Y/%m/%d')
    #            for d in days]
    # convert to dataframe and write to excel
    df = pd.DataFrame(all_recon_tasks)

    writer = pd.ExcelWriter('D:/j0126_analysis/reconnect_stats.xls')
    df.to_excel(writer, 'reconnects')
    workbook = writer.book
    worksheet = writer.sheets['reconnects']
    worksheet.set_zoom(90)

    hours_format = workbook.add_format(
        {'num_format': '0.00', 'bold': True})

    worksheet.set_column('A:A', 20)
    worksheet.set_column('B:Z', 20, hours_format)
    writer.save()

    # dataframe with all individual reconnects as rows, entries:
    # time
    # skel length
    # path to knossos skeleton

    # numbers:
    # total hours spend
    total_hours = sum(all_recon_tasks['time [s]']) / 3600.

    print('Total hours: {0}'.format(total_hours))
    # num annotators
    # mean time no reconnect
    # mean time with skeleton
    # examples of long tracings, short tracings and no tracings

    # dataframe with
    make_plots = True

    if make_plots:
        #plt.figure()
        #for this_skel_ts in all_timestamps:
        #    plt.plot(this_skel_ts, [1.]*len(this_skel_ts))
        #plt.title('all node timestamps')
        #plt.xlabel('ts in s')

        plt.figure()
        plt.hist([t for t in all_recon_tasks['time [s]'] if t < 500], bins=100)
        plt.xlabel('t [s]')
        plt.title('hist of total times')
        plt.xlim([0,500])

        plt.figure()
        plt.hist([t for t in all_recon_tasks['time skel [s]'] if t < 500 and t > 0.1], bins=100)
        plt.xlabel('t [s]')
        plt.title('hist of skeleton times')
        plt.xlim([0,500])

        plt.figure()
        plt.hist([l for l in all_recon_tasks['length [um]'] if l < 50 and l > 0.1], bins=100)
        plt.xlabel('skel length [um]')
        plt.title('hist of skeleton lengths')
        plt.xlim([0,50])

        #plt.figure()
        #plt.hist(skel_times, bins=100)
        #plt.xlabel('skel time [s]')
        #plt.title('hist of skeleton times')


    return


if __name__ == "__main__":
    analyze_all_j0126_reconnector_tasks()