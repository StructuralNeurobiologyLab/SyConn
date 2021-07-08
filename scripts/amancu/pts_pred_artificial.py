import glob
import argparse
import numpy as np
import torch
import os
from torch import nn
from tqdm import tqdm
import pandas as pd
from morphx.classes.hybridcloud import HybridCloud
import morphx.processing.clouds as clouds
from morphx.processing.hybrids import extract_subset
from morphx.processing.objects import context_splitting_graph_many
from elektronn3.models.convpoint import SegSmall
from elektronn3.models.lcp_adapt import ConvAdaptSeg
from lightconvpoint.utils.network import get_search, get_conv
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc
from scipy.special import softmax
from matplotlib import pyplot
from syconn.proc.meshes import mesh2obj_file_colors

def prob(pred):
    # determine indices
    ind = np.argmax(pred,1).astype(np.float64)
    ind = ind.astype(np.int64)
    # normalize predictions
    softened = softmax(pred,axis=1)
    # print(softened)
    res = []
    for i in range(len(pred)):
        nr = softened[i]
        nr = nr[ind[i]]
        res.append(nr if ind[i]==1 else 1-nr)
    res = np.array(res)
    # print(res)
    return res



# radii = [100, 500, 1000, 2000, 5000]
radii = [2000]
# radii = [5000]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction pipeline for merge error detection')
    parser.add_argument('--r', type=int, help='Radius of cs merger',
                        default=100)
    args = parser.parse_args()
    radius = args.r

    # colors for labels
    RED = np.array([255., 125., 125., 255.])
    GREY = np.array([180., 180., 180., 255.])

    # define model args
    input_channels = 1
    num_classes = 2
    use_norm = 'gn'
    dr = 0.2
    track_running_stats = False
    act = 'relu'
    use_bias = True
    scale_norm=5e3
    ctx_size=20e3
    pred_transform = clouds.Compose([clouds.Center(), clouds.Normalization(scale_norm)])
    # pred_transform = clouds.Compose([clouds.RandomVariation((-30, 30), distr='normal'),  # in nm
    #                               clouds.Center(),
    #                               clouds.Normalization(scale_norm),
    #                               clouds.RandomRotate(apply_flip=True),
    #                               clouds.ElasticTransform(res=(40, 40, 40), sigma=6),
    #                               clouds.RandomScale(distr_scale=0.1, distr='uniform')])


    lcp_flag = True

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    for radius in radii:
        # test set
        # folder = f'/wholebrain/scratch/amancu/mergeError/test_dataset/R{radius}/*.pkl'
        # training set
        folder = f'/wholebrain/scratch/amancu/mergeError/ptclouds/R{radius}/Hybridcloud/*.pkl'
        if lcp_flag:
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/mergeError_pts_model_lcp_radius{radius}_eval0_ConvPoint_SearchQuantized/state_dict.pth'
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/mergeError_pts_model_lcp_radius{radius}_ConvPoint_SearchQuantized_Adam/state_dict.pth'
            save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/mergeError_pts_model_lcp_radius{radius}_ConvPoint_SearchQuantized_Adam_weights1,2/state_dict.pth'
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/mergeError_pts_model_lcp_radius{radius}_eval0_FKAConv_SearchQuantized/state_dict.pth'
        pred_files = glob.glob(folder)

        print(f'Predicting for radius {radius} with nr of files {len(pred_files)}')

        with torch.no_grad():
            if lcp_flag:
                search = 'SearchQuantized'
                conv = dict(layer='ConvPoint', kernel_separation=False)
                # conv = dict(layer='FKAConv', kernel_separation=False)
                act = torch.nn.ReLU
                model = ConvAdaptSeg(input_channels, num_classes, get_conv(conv), get_search(search), kernel_num=64,
                                     architecture=None, activation=act, norm='gn').to(device)
            else:
                model = SegSmall(input_channels, num_classes + 1, dropout=dr, use_norm=use_norm,
                                track_running_stats=track_running_stats, act=act, use_bias=use_bias).to(device)
            model.load_state_dict(torch.load(save_path, map_location=device)['model_state_dict'])
            # print(model)
            model.eval()

            hc = HybridCloud()
            result_dc = collections.defaultdict(list)

            # for each whole cells
            for i in tqdm(range(len(pred_files))):
                path = pred_files[i]
                hc.load_from_pkl(path)

                pred_transform(hc)

                sample_pts = hc.vertices
                sample_pts = sample_pts[None,:,:]
                sample_feats = hc.features
                sample_feats = sample_feats[None, :, None]
                # print(f' hc feats shape {hc.features.shape}, sub feats shape {sample_feats.shape}')
                sample_labels = hc.labels

                dpts = torch.from_numpy(sample_pts).to(device).float()
                dfeats = torch.from_numpy(sample_feats).to(device).float()

                if lcp_flag:
                    dpts = dpts.transpose(1, 2)
                    dfeats = dfeats.transpose(1, 2)

                # print(f'pts shape {dpts.shape} feats shape {dfeats.shape}')

                with torch.no_grad():
                    try:
                        pred = model(dfeats, dpts)
                    except:
                        print('No se puede')
                        continue
                    if lcp_flag:
                        pred = pred.transpose(1, 2)
                    pred = pred.detach().cpu().numpy()
                    # eliminate batch axis
                    pred = pred[0,:,:]
                    # print(pred)

                probs = prob(pred)
                # prepare predictions
                pred = np.argmax(pred,1).astype(np.float64)
                # print(np.unique(pred))
                # evaluate result
                precision = precision_score(sample_labels, pred, average='binary', zero_division=0)

                # print(precision)
                recall = recall_score(sample_labels, pred, average='binary', zero_division=0)
                # print(recall)
                fscore = f1_score(sample_labels, pred, average='binary', zero_division=0)
                # print(fscore)

                lr_precision, lr_recall, _ = precision_recall_curve(sample_labels, probs)
                # print(f'Normal precision {precision} and recall {recall}')
                # print(f'Alternative precision {lr_precision} and recall {lr_recall}')
                aucc = auc(lr_recall, lr_precision)

                result_dc['precision'].append(precision)
                result_dc['recall'].append(recall)
                result_dc['fscore'].append(fscore)
                result_dc['auc'].append(auc)

                # print(f'Fscore: {fscore} and auc: {aucc} for cell merger {os.path.basename(pred_files[i])}')

                # plot_precision_recall(sample_labels, pred)

                # #original
                # colors = np.full(shape=(sample_pts.shape[1],4, ), fill_value=GREY)
                # mask = np.where(hc.labels == 1)[0]
                # mask = np.array([[x] for x in mask])
                # try:
                #     np.put_along_axis(colors, mask, RED, axis=0)
                # except:
                #     print("No foreground labels in original context.")
                # mesh2obj_file_colors(os.path.expanduser(
                #     f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/{radius}/Adam_stepLR/' + os.path.basename(pred_files[i]) + f'_original.ply'),
                #     [np.array([]), hc.vertices, np.array([])], colors)
                #
                # # prediction
                # colors = np.full(shape=(sample_pts.shape[1],4, ), fill_value=GREY)
                # mask = np.where(pred == 1)[0].astype(np.int64)
                # mask = np.array([[x] for x in mask])
                # try:
                #     np.put_along_axis(colors, mask, RED, axis=0)
                # except:
                #     print("No foreground labels in prediction.")
                # mesh2obj_file_colors(os.path.expanduser(
                #     f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/{radius}/Adam_stepLR/' + os.path.basename(pred_files[i]) + f'_prediction.ply'),
                #     [np.array([]), hc.vertices, np.array([])], colors)

            print(f'Number zero Fscores: {len(np.where(result_dc.fscore==0.0))}')
            print(f'Mean fscores: {np.mean(result_dc.fscore)}')
            # print(f'Aucs: {aucs}')
            print(f'Mean aucs: {np.mean(result_dc.auc)}')

            df = pd.DataFrame()

            # proceed to save this to dict




            # # for each whole cells
            # for i in range(20,25):
            #     path = pred_files[i]
            #     hc.load_from_pkl(path)
            #
            #     source_nodes = np.array([])
            #     for label in (0, 1):
            #         if source_nodes.size == 0:
            #             source_nodes = np.where(hc.node_labels == label)[0]
            #         else:
            #             source_nodes = np.concatenate((source_nodes, np.where(hc.node_labels == label)[0]))
            #
            #     source_nodes = np.random.choice(source_nodes, batch_size)
            #     print(len(source_nodes))
            #
            #     # create contexts from whole cells
            #     for ii, source_node in enumerate(source_nodes):
            #         if np.random.randint(0, 4) == 0:
            #             fluct = 1
            #         else:
            #             fluct = min(max(np.random.randn(1)[0] * 0.1 + 1, 0.8), 1.2)
            #         ctx_size_fluct = fluct * ctx_size
            #
            #         # extract context
            #         node_ids = context_splitting_graph_many(hc, [source_node], ctx_size_fluct)[0]
            #         hc_sub = extract_subset(hc, node_ids)[0]  # only pass HybridCloud
            #
            #         pred_transform(hc_sub)
            #
            #         sample_pts = hc_sub.vertices
            #         sample_pts = sample_pts[None,:,:]
            #         sample_feats = hc_sub.features
            #         sample_feats = sample_feats[None, :, None]
            #         # print(f' hc feats shape {hc.features.shape}, sub feats shape {sample_feats.shape}')
            #         sample_labels = hc_sub.labels
            #
            #         dpts = torch.from_numpy(sample_pts).to(device).float()
            #         dfeats = torch.from_numpy(sample_feats).to(device).float()
            #
            #         if lcp_flag:
            #             dpts = dpts.transpose(1, 2)
            #             dfeats = dfeats.transpose(1, 2)
            #
            #         print(f'pts shape {dpts.shape} feats shape {dfeats.shape}')
            #
            #         with torch.no_grad():
            #             try:
            #                 pred = model(dfeats, dpts)
            #             except:
            #                 print('No se puede')
            #                 continue
            #             if lcp_flag:
            #                 pred = pred.transpose(1, 2)
            #             pred = pred.detach().cpu().numpy()
            #             # eliminate batch axis
            #             pred = pred[0,:,:]
            #             # print(pred.shape)
            #
            #         # prepare predictions
            #         pred = np.argmax(pred,1).astype(np.float64)
            #         # print(np.unique(pred))
            #         # evaluate result
            #         precision = precision_score(sample_labels, pred, average='binary')
            #         # print(precision)
            #         recall = recall_score(sample_labels, pred, average='binary', zero_division=0)
            #         # print(recall)
            #         fscore = f1_score(sample_labels, pred, average='binary')
            #         # print(fscore)
            #
            #         lr_precision, lr_recall, _ = precision_recall_curve(sample_labels, pred)
            #         print(f'Normal precision {precision} and recall {recall}')
            #         print(f'Alternative precision {lr_precision} and recall {lr_recall}')
            #         try:
            #             auc = auc(lr_recall, lr_precision)
            #         except:
            #             pass
            #
            #         print(f'Fscore: {fscore} and auc: {auc} for cell merger {os.path.basename(pred_files[i])}_{ii}')
            #
            #         # plot_precision_recall(sample_labels, pred)
            #
            #         #original
            #         colors = np.full(shape=(sample_pts.shape[1],4, ), fill_value=GREY)
            #         mask = np.where(hc_sub.labels == 1)[0]
            #         mask = np.array([[x] for x in mask])
            #         try:
            #             np.put_along_axis(colors, mask, RED, axis=0)
            #         except:
            #             print("No foreground labels in original context.")
            #         mesh2obj_file_colors(os.path.expanduser(
            #             f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/{radius}/Adam_stepLR/' + os.path.basename(pred_files[i]) + f'_original_{ii}.ply'),
            #             [np.array([]), hc_sub.vertices, np.array([])], colors)
            #
            #         # prediction
            #         colors = np.full(shape=(sample_pts.shape[1],4, ), fill_value=GREY)
            #         mask = np.where(pred == 1)[0].astype(np.int64)
            #         mask = np.array([[x] for x in mask])
            #         try:
            #             np.put_along_axis(colors, mask, RED, axis=0)
            #         except:
            #             print("No foreground labels in prediction.")
            #         mesh2obj_file_colors(os.path.expanduser(
            #             f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/{radius}/Adam_stepLR/' + os.path.basename(pred_files[i]) + f'_prediction_{ii}.ply'),
            #             [np.array([]), hc_sub.vertices, np.array([])], colors)