import pytorch_lightning as pl
import cv2
import numpy as np
from model import Model
from config import cfg  # Assuming the configuration is defined in config.py
import torch 
from pytorch_lightning.callbacks import ModelCheckpoint
from train import VoxelNetPL
from sklearn.preprocessing import LabelEncoder
from data import create_data_loader
# Define the parameters and device
from utils.utils import box3d_center_to_corner_batch, cal_anchors, delta_to_boxes3d
def visualize():
    params = params = {
            "n_epochs":     15,
            "batch_size":   2,  # Set to 2 for the fastest training. (Training time exponentially grows with batch size). Batch Size > 16 causes GPU memory issues for me.
            "small_addon_for_BCE": 1e-6,
            "max_gradient_norm": 5, 
            "alpha_bce":    1.5,
            "beta_bce":     1.0,
            "learning_rate": 0.001,
            "mode":         "train",
            "dump_vis":     "no",
            "data_root_dir": "../",
            "model_dir":    "model",
            "model_name":   "model1",
            "num_threads":  8
        }
    cfg["DATA_DIR"] = params["data_root_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_instance = Model(cfg, params, device).to(device)
    loaded_model = VoxelNetPL.load_from_checkpoint(
        "./model/epoch=100-step=302293.ckpt",
        model=model_instance
    ).to(device)
    loaded_model.eval()
    label_encoder = LabelEncoder()
    _    , val_loader = create_data_loader(cfg, params, 16, "train", is_aug_data=True, label_encoder=label_encoder, create_anchors=True, seed=2023)
    val_iter = iter(val_loader)

    response = None
    while response == None:
        data = next(val_iter)
        feature_buffer      = data["feature_buffer"].to(device)
        coordinate_buffer   = data["coordinate_buffer"].to(device)
        targets             = data["targets"].to(device)
        pos_equal_one       = data["pos_equal_one"].to(device)
        images              = data["img"]
        calib               = data["calib"]
        tags                = data["tag"]
        r_map, p_map = loaded_model.model(feature_buffer, coordinate_buffer)
        response = draw_boxes(p_map.permute(0, 3, 1, 2).detach(), r_map.permute(0, 3, 1, 2).detach(), images, calib, tags)
        # response = draw_boxes(targets, pos_equal_one, images, calib,  tags) 
def project_velo2rgb(velo,calib):
    T=np.zeros([4,4],dtype=np.float32)
    T[:3,:]=calib['Tr_velo2cam']
    T[3,3]=1
    R=np.zeros([4,4],dtype=np.float32)
    R[:3,:3]=calib['R0']
    R[3,3]=1
    num=len(velo)
    projections = np.zeros((num,8,2),  dtype=np.int32)
    for i in range(len(velo)):
        box3d=np.ones([8,4],dtype=np.float32)
        box3d[:,:3]=velo[i]
        M=np.dot(calib['P2'],R)
        M=np.dot(M,T)
        box2d=np.dot(M,box3d.T)
        box2d=box2d[:2,:].T/box2d[2,:].reshape(8,1)
        projections[i] = box2d
    return projections

def draw_rgb_projections(image, projections, color=(255,255,255), thickness=5, darker=1):

    img = image.copy()*darker
    num=len(projections)
    forward_color=(255,255,0)
    for n in range(num):
        qs = projections[n]
        for k in range(0,4):
            i,j=k,(k+1)%4
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

            i,j=k+4,(k+1)%4 + 4
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

            i,j=k,k+4
            cv2.line(img, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

        cv2.line(img, (qs[3,0],qs[3,1]), (qs[7,0],qs[7,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[7,0],qs[7,1]), (qs[6,0],qs[6,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[6,0],qs[6,1]), (qs[2,0],qs[2,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[2,0],qs[2,1]), (qs[3,0],qs[3,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[3,0],qs[3,1]), (qs[6,0],qs[6,1]), forward_color, thickness, cv2.LINE_AA)
        cv2.line(img, (qs[2,0],qs[2,1]), (qs[7,0],qs[7,1]), forward_color, thickness, cv2.LINE_AA)

    return img

# def delta_to_boxes3d(deltas, anchors):
#     # Input:
#     #   deltas: (N, w, l, 14)
#     #   feature_map_shape: (w, l)
#     #   anchors: (w, l, 2, 7)

#     # Ouput:
#     #   boxes3d: (N, w*l*2, 7)
#     N = deltas.shape[0]
#     deltas = deltas.view(N, -1, 7)
#     anchors = torch.FloatTensor(anchors)
#     boxes3d = torch.zeros_like(deltas)

#     if deltas.is_cuda:
#         anchors = anchors.cuda()
#         boxes3d = boxes3d.cuda()

#     anchors_reshaped = anchors.view(-1, 7)

#     anchors_d = torch.sqrt(anchors_reshaped[:, 4]**2 + anchors_reshaped[:, 5]**2)

#     anchors_d = anchors_d.repeat(N, 2, 1).transpose(1,2)
#     anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

#     boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + anchors_reshaped[..., [0, 1]]
#     boxes3d[..., [2]] = torch.mul(deltas[..., [2]], anchors_reshaped[...,[3]]) + anchors_reshaped[..., [2]]

#     boxes3d[..., [3, 4, 5]] = torch.exp(
#         deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]

#     boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

#     return boxes3d



def draw_boxes(reg, prob, images, calibs, tags):
    
    response = None
    prob = prob.view(2, -1)
    batch_boxes3d = delta_to_boxes3d(reg)
    score_threshold = 0.5
    mask = torch.gt(prob, score_threshold)
    mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

    for batch_id in range(2):
        boxes3d = torch.masked_select(batch_boxes3d[batch_id], mask_reg[batch_id]).view(-1, 7)
        # scores = torch.masked_select(prob[batch_id], mask[batch_id])
        image = images[batch_id].cpu().numpy()
        image = (image * 255).astype(np.uint8)
        calib = {
            "P2": calibs["P2"][batch_id],
            "R0": calibs["R0"][batch_id],
            "Tr_velo2cam": calibs["Tr_velo2cam"][batch_id]
        }
        tag = tags[batch_id]
        if len(boxes3d) != 0:

            boxes3d_corner = box3d_center_to_corner_batch(boxes3d.cpu().numpy())
            # boxes2d = box3d_corner_to_top_batch(boxes3d_corner)
            # boxes2d_score = torch.cat((boxes2d, scores.unsqueeze(1)), dim=1)

            # # NMS
            # keep = pth_nms(boxes2d_score, cfg.nms_threshold)
            # boxes3d_corner_keep = boxes3d_corner[keep]
            print("No. %d objects detected" % len(boxes3d_corner))

            rgb_2D = project_velo2rgb(boxes3d_corner, calib)
            img_with_box = draw_rgb_projections(image.transpose(1,2,0), rgb_2D, color=(255,192,203), thickness=5)
            # img_with_box = cv2.resize(img_with_box, (img_with_box.shape[1] // 2, img_with_box.shape[2] // 2))
            cv2.imwrite(f'results/{tag}.png', img_with_box)
            response = "Success"

        else:
            print("No objects detected")
        return response

if __name__ == '__main__':
    visualize()