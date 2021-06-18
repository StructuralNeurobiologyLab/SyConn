from syconn.handler.prediction_pts import predict_pts_plain, pts_loader_scalar, pts_pred_scalar
from syconn.reps.super_segmentation import SuperSegmentationObject, SuperSegmentationDataset
from morphx.classes.pointcloud import PointCloud
import torch
import numpy as np

# define model args
device = torch.device('cuda')
input_channels = 1
num_classes = 2
use_norm = 'gn'
dr = 0.2
track_running_stats = False
act = 'relu'
use_bias = True
npoints = 25000
scale_fact = 5000

def load_model(mkwargs, device):
    from elektronn3.models.convpoint import SegSmall
    m = SegSmall(input_channels, num_classes + 1, dropout=dr, use_norm=use_norm,
                             track_running_stats=track_running_stats, act=act, use_bias=use_bias).to(device)
    m.load_state_dict(torch.load(mdir)['model_state_dict'])
    m = torch.nn.DataParallel(m)
    m.eval()
    return m

radius = 100
ssv_ids = [224145260, 351931151, 205268369, 81874455, 316966179, 271546135, 111149122, 2972211, 81874455]

ssd_kwargs = dict(working_dir='/ssdscratch/songbird/j0251/rag_flat_Jan2019_v3')
ssd = SuperSegmentationDataset(**ssd_kwargs)
mdir = f'/wholebrain/scratch/amancu/mergeError/trainings/mergeError_pts_model_segsmall_radius{radius}_cellshapeOnly_gn_eval0/state_dict.pth'
mkwargs = dict(use_bn=False, track_running_stats=False)
ssd_kwargs = [{'ssv_id': ssv_id, 'working_dir': ssd_kwargs['working_dir']} for ssv_id in ssv_ids]
model = load_model(mkwargs, device)
model.eval()

# get SSOs and predict
for ssv_id in ssv_ids:
    sso = SuperSegmentationObject(ssv_id)
    verts = sso.mesh[1].reshape((-1, 3))
    feats = np.ones(shape=(len(verts),))
    dverts = torch.from_numpy(verts).to(device).float()
    dfeats = torch.from_numpy(feats).to(device).float()
    with torch.no_grad():
        pred = model(dfeats, dverts)

    pred = pred.detach().cpu().numpy()


#dict_out = predict_pts_plain(ssd_kwargs, load_model, pts_loader_scalar, pts_pred_scalar,
#                            npoints, scale_fact, ssv_ids=ssv_ids,
#                             nloader=2, npredictor=1, use_test_aug=True, ctx_size=20000)