# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import cv2
import numpy as np
import imageio
import torch
import flow_vis

from matplotlib import cm
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from .utils import bivariate_Gaussian


def read_video_from_path(path):
    try:
        reader = imageio.get_reader(path)
    except Exception as e:
        print("Error opening video file: ", e)
        return None
    frames = []
    for i, im in enumerate(reader):
        frames.append(np.array(im))
    return np.stack(frames)


def draw_circle(rgb, coord, radius, color=(255, 0, 0), visible=True):
    # Create a draw object
    draw = ImageDraw.Draw(rgb)
    # Calculate the bounding box of the circle
    left_up_point = (coord[0] - radius, coord[1] - radius)
    right_down_point = (coord[0] + radius, coord[1] + radius)
    # Draw the circle
    if not visible:
        return rgb

    draw.ellipse(
        [left_up_point, right_down_point],
        fill=tuple(color) if visible else None,
        outline=tuple(color),
    )
    return rgb


def draw_line(rgb, coord_y, coord_x, color, linewidth):
    draw = ImageDraw.Draw(rgb)
    draw.line(
        (coord_y[0], coord_y[1], coord_x[0], coord_x[1]),
        fill=tuple(color),
        width=linewidth,
    )
    return rgb


def add_weighted(rgb, alpha, original, beta, gamma):
    return (rgb * alpha + original * beta + gamma).astype("uint8")


########## tracks visualization ########## 

class Visualizer:
    def __init__(
        self,
        save_dir: str = "./results",
        grayscale: bool = False,
        pad_value: int = 0,
        fps: int = 10,
        mode: str = "rainbow",  # 'cool', 'optical_flow'
        linewidth: int = 2,
        show_first_frame: int = 1,
        tracks_leave_trace: int = 0,  # -1 for infinite
    ):
        self.mode = mode
        self.save_dir = save_dir
        if mode == "rainbow":
            self.color_map = cm.get_cmap("gist_rainbow")
        if mode == 'rainbow_enhanced':
            self.color_map = cm.get_cmap("gist_rainbow")
        elif mode == 'rainbow_all':
            self.color_map = None
        elif mode == "cool":
            self.color_map = cm.get_cmap(mode)
        self.show_first_frame = show_first_frame
        self.grayscale = grayscale
        self.tracks_leave_trace = tracks_leave_trace
        self.pad_value = pad_value
        self.linewidth = linewidth
        self.fps = fps

        self.blur_kernel = bivariate_Gaussian(kernel_size=99, sig_x=10, sig_y=10, theta=0, grid=None, isotropic=True)

    def visualize(
        self,
        video: torch.Tensor,  # (B,T,C,H,W)
        tracks: torch.Tensor,  # (B,T,N,2)
        visibility: torch.Tensor = None,  # (B, T, N, 1) bool
        gt_tracks: torch.Tensor = None,  # (B,T,N,2)
        segm_mask: torch.Tensor = None,  # (B,1,H,W)
        filename: str = "video",
        writer=None,  # tensorboard Summary Writer, used for visualization during training
        step: int = 0,
        query_frame: int = 0,
        save_video: bool = True,
        compensate_for_camera_motion: bool = False,
        rigid_part = None,
        video_depth = None, # (B,T,C,H,W)
        depths = None,
        circle_scale=1,
        is_blur=False,
        is_depth_norm=False,
    ):
        if compensate_for_camera_motion:
            assert segm_mask is not None
        if segm_mask is not None:
            coords = tracks[0, query_frame].round().long()
            segm_mask = segm_mask[0, query_frame][coords[:, 1], coords[:, 0]].long()

        video = F.pad(
            video,
            (self.pad_value, self.pad_value, self.pad_value, self.pad_value),
            "constant",
            255,
        )

        if video_depth is not None:
            video_depth = (video_depth*255).cpu().numpy().astype(np.uint8)
            video_depth = ([cv2.applyColorMap(video_depth[0,i,0], cv2.COLORMAP_INFERNO) 
                            for i in range(video_depth.shape[1])])
            video_depth = np.stack(video_depth, axis=0)
            video_depth = torch.from_numpy(video_depth).permute(0, 3, 1, 2)[None]

        tracks = tracks + self.pad_value

        if self.grayscale:
            transform = transforms.Grayscale()
            video = transform(video)
            video = video.repeat(1, 1, 3, 1, 1)
        
        res_video = self.draw_tracks_on_video(
            video=video,
            tracks=tracks,
            visibility=visibility,
            segm_mask=segm_mask,
            gt_tracks=gt_tracks,
            query_frame=query_frame,
            compensate_for_camera_motion=compensate_for_camera_motion,
            rigid_part=rigid_part,
            depths=depths,
            circle_scale=circle_scale,
            is_blur=is_blur,
            is_depth_norm=is_depth_norm,
        )
        
        if save_video:
            self.save_video(res_video, filename=filename, 
                            writer=writer, step=step)
            if video_depth is not None:
                self.save_video(video_depth, filename=filename+"_depth", 
                                writer=writer, step=step)
        return res_video

    def save_video(self, video, filename, writer=None, step=0):
        if writer is not None:
            writer.add_video(
                f"{filename}_pred_track",
                video.to(torch.uint8),
                global_step=step,
                fps=self.fps,
            )
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            wide_list = list(video.unbind(1))
            wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]

            # Prepare the video file path
            save_path = os.path.join(self.save_dir, f"{filename}.mp4")

            # Create a writer object
            video_writer = imageio.get_writer(save_path, fps=self.fps)

            # Write frames to the video file
            for frame in wide_list:
                video_writer.append_data(frame)

            video_writer.close()
            print(f"Video saved to {save_path}")


    def draw_tracks_on_video(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor = None,
        segm_mask: torch.Tensor = None,
        gt_tracks=None,
        query_frame: int = 0,
        compensate_for_camera_motion=False,
        rigid_part=None,
        depths=None,
        circle_scale=1,
        is_blur=False,
        is_depth_norm=False,
    ):
        B, T, C, H, W = video.shape
        _, _, N, D = tracks.shape

        assert D == 2
        assert C == 3
        video = video[0].permute(0, 2, 3, 1).byte().detach().cpu().numpy()  # S, H, W, C
        tracks = tracks[0].long().detach().cpu().numpy()  # S, N, 2
        if gt_tracks is not None:
            gt_tracks = gt_tracks[0].detach().cpu().numpy()

        res_video = []

        # process input video
        for rgb in video:
            res_video.append(rgb.copy())

        vector_colors = np.zeros((T, N, 3))
        if self.mode == "optical_flow":
            vector_colors = flow_vis.flow_to_color(tracks - tracks[query_frame][None])
        elif segm_mask is None:
            if self.mode == "rainbow":
                y_min, y_max = (
                    tracks[query_frame, :, 1].min(),
                    tracks[query_frame, :, 1].max(),
                )
                norm = plt.Normalize(y_min, y_max)
                for n in range(N):
                    color = self.color_map(norm(tracks[query_frame, n, 1]))
                    color = np.array(color[:3])[None] * 255
                    vector_colors[:, n] = np.repeat(color, T, axis=0)
            elif self.mode == 'rainbow_all':
                a = np.ones([H, W, 3])
                for i in range(a.shape[1]):
                    for j in range(a.shape[0]):
                        a[j, i, 0] = i / W * 180
                        a[j, i, 1] = 250
                        a[j, i, 2] = j / H * 180 + 50
                a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_HSV2RGB)
                for n in range(N):
                    color = a[tracks[query_frame, n, 1], tracks[query_frame, n, 0]]
                    color = np.array(color[:3])[None]
                    vector_colors[:, n] = np.repeat(color, T, axis=0)
            else:
                # color changes with time
                for t in range(T):
                    color = np.array(self.color_map(t / T)[:3])[None] * 255
                    vector_colors[t] = np.repeat(color, N, axis=0)
        else:
            if self.mode == "rainbow":
                vector_colors[:, segm_mask <= 0, :] = 255

                y_min, y_max = (
                    tracks[0, segm_mask > 0, 1].min(),
                    tracks[0, segm_mask > 0, 1].max(),
                )
                norm = plt.Normalize(y_min, y_max)
                for n in range(N):
                    if segm_mask[n] > 0:
                        color = self.color_map(norm(tracks[0, n, 1]))
                        color = np.array(color[:3])[None] * 255
                        vector_colors[:, n] = np.repeat(color, T, axis=0)

            else:
                # color changes with segm class
                segm_mask = segm_mask.cpu()
                color = np.zeros((segm_mask.shape[0], 3), dtype=np.float32)
                color[segm_mask > 0] = np.array(self.color_map(1.0)[:3]) * 255.0
                color[segm_mask <= 0] = np.array(self.color_map(0.0)[:3]) * 255.0
                vector_colors = np.repeat(color[None], T, axis=0)

        #  draw tracks
        if self.tracks_leave_trace != 0:
            for t in range(1, T):
                first_ind = (
                    max(0, t - self.tracks_leave_trace)
                    if self.tracks_leave_trace >= 0
                    else 0
                )
                curr_tracks = tracks[first_ind : t + 1]
                curr_colors = vector_colors[first_ind : t + 1]
                if compensate_for_camera_motion:
                    diff = (
                        tracks[first_ind : t + 1, segm_mask <= 0]
                        - tracks[t : t + 1, segm_mask <= 0]
                    ).mean(1)[:, None]

                    curr_tracks = curr_tracks - diff
                    curr_tracks = curr_tracks[:, segm_mask > 0]
                    curr_colors = curr_colors[:, segm_mask > 0]

                res_video[t] = self._draw_pred_tracks(
                    res_video[t],
                    curr_tracks,
                    curr_colors,
                )
                if gt_tracks is not None:
                    res_video[t] = self._draw_gt_tracks(
                        res_video[t], gt_tracks[first_ind : t + 1]
                    )

        if rigid_part is not None:
            cls_label = torch.unique(rigid_part)
            cls_num = len(torch.unique(rigid_part))
            # visualize the clustering results 
            cmap = plt.get_cmap('jet')  # get the color mapping
            colors = cmap(np.linspace(0, 1, cls_num))  
            colors = (colors[:, :3] * 255) 
            color_map = {lable.item(): color for lable, color in zip(cls_label, colors)}

        # depth visualization
        if depths is not None:
            if not is_depth_norm:
                if depths.max() != depths.min():
                    depths = 1.1 - (depths - depths.min()) / (depths.max() - depths.min())
            else:
                depths = 1.1 - depths
            depths = depths.squeeze(0)

        #  draw points
        for t in range(T):
            img = Image.fromarray(np.uint8(res_video[t]))
            indices = np.argsort(depths[t])
            coords, depth, pcolors = tracks[t][indices], depths[t][indices], vector_colors[t][indices]
            for i in range(N):
                # coord = (tracks[t, i, 0], tracks[t, i, 1])
                coord = (coords[i, 0], coords[i, 1])
                visibile = True
                if visibility is not None:
                    visibile = visibility[0, t, i]
                if coord[0] != 0 and coord[1] != 0:
                    if not compensate_for_camera_motion or (
                        compensate_for_camera_motion and segm_mask[i] > 0
                    ):
                        if rigid_part is not None:
                            color = color_map[rigid_part.squeeze()[i].item()]
                            img = draw_circle(
                                img,
                                coord=coord,
                                radius=int(self.linewidth * 2) if depths is None else int(self.linewidth * circle_scale * depth[i]),
                                color=color.astype(int),
                                visible=visibile,
                            )
                        else:
                            if visibile:
                                img = draw_circle(
                                    img,
                                    coord=coord,
                                    radius=int(self.linewidth * 2) if depths is None else int(self.linewidth * circle_scale * depth[i]),
                                    color=pcolors[i].astype(int),
                                    visible=visibile,
                                )
                        if is_blur:
                            res_video[t] = cv2.filter2D(res_video[t], -1, self.blur_kernel)
                        else:
                            res_video[t] = np.array(img)

        #  construct the final rgb sequence
        if self.show_first_frame > 0:
            res_video = [res_video[0]] * self.show_first_frame + res_video[1:]
        return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte()

    def _draw_pred_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3
        tracks: np.ndarray,  # T x 2
        vector_colors: np.ndarray,
        alpha: float = 0.5,
    ):
        T, N, _ = tracks.shape

        for s in range(T - 1):
            vector_color = vector_colors[s]
            original = rgb.copy()
            alpha = (s / T) ** 2
            for i in range(N):
                coord_y = (int(tracks[s, i, 0]), int(tracks[s, i, 1]))
                coord_x = (int(tracks[s + 1, i, 0]), int(tracks[s + 1, i, 1]))
                if coord_y[0] != 0 and coord_y[1] != 0:
                    cv2.line(
                        rgb,
                        coord_y,
                        coord_x,
                        vector_color[i].tolist(),
                        self.linewidth,
                        cv2.LINE_AA,
                    )
            if self.tracks_leave_trace > 0:
                rgb = cv2.addWeighted(rgb, alpha, original, 1 - alpha, 0)
        return rgb

    def _draw_gt_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3,
        gt_tracks: np.ndarray,  # T x 2
    ):
        T, N, _ = gt_tracks.shape
        color = np.array((211.0, 0.0, 0.0))

        for t in range(T):
            for i in range(N):
                gt_tracks = gt_tracks[t][i]
                #  draw a red cross
                if gt_tracks[0] > 0 and gt_tracks[1] > 0:
                    length = self.linewidth * 3
                    coord_y = (int(gt_tracks[0]) + length, int(gt_tracks[1]) + length)
                    coord_x = (int(gt_tracks[0]) - length, int(gt_tracks[1]) - length)
                    cv2.line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth,
                        cv2.LINE_AA,
                    )
                    coord_y = (int(gt_tracks[0]) - length, int(gt_tracks[1]) + length)
                    coord_x = (int(gt_tracks[0]) + length, int(gt_tracks[1]) - length)
                    cv2.line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth,
                        cv2.LINE_AA,
                    )
        return rgb


########## optical flow visualization ########## 

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


def vis_flow_to_video(optical_flow, num_frames):
    '''
    optical_flow: T x H x W x 2
    '''
    video = []
    for i in range(0, num_frames):
        flow_img = flow_to_image(optical_flow[i])
        flow_img = torch.Tensor(flow_img) # H x W x 3
        video.append(flow_img)
    video = torch.stack(video, dim=0) # T x H x W x 3
    return video


def vis_depth_to_video(depth, num_frames):
    '''
    depth: T x H x W x 1
    '''
    video = []
    for i in range(0, num_frames):
        depth_img = colorize(depth[i])
        depth_img = torch.Tensor(depth_img) # H x W x 3
        video.append(depth_img)
    video = torch.stack(video, dim=0) # T x H x W x 3
    return video


def colorize(
    value: np.ndarray, vmin: float = None, vmax: float = None, cmap: str = "magma_r"
):
    # if already RGB, do nothing
    if value.ndim > 2:
        if value.shape[-1] > 1:
            return value
        value = value[..., 0]
    invalid_mask = value < 0.0001
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # set color
    cmapper = plt.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 0
    img = value[..., :3]
    return img

# from https://github.com/gengshan-y/VCN
def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx]) # 光流越小，颜色越亮。这样可以使得静止或者运动较慢的区域在可视化结果中更加明显
        notidx = np.logical_not(idx) 

        col[notidx] *= 0.75 # 光流越大，颜色越暗
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel