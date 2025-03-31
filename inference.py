import os
import cv2
import json
import numpy as np
import gradio as gr
from PIL import Image
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from diffusers import AutoencoderKL, DDIMScheduler
from einops import repeat
from omegaconf import OmegaConf
from decord import VideoReader
from transformers import CLIPVisionModelWithProjection

from src.models.motion_encoder import MotionEncoder
from src.models.fusion_module import FusionModule
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_motion2vid_merge_infer import Motion2VideoPipeline
from src.utils.util import save_videos_grid
from src.utils.utils import interpolate_trajectory, interpolate_trajectory_3d
from src.utils.visualizer import Visualizer

import safetensors
    

def visualize_tracks(background_image_path, splited_tracks, video_length, width, height, save_path='./tmp/hint.mp4'):
    
    background_image = Image.open(background_image_path).convert('RGBA')
    background_image = background_image.resize((width, height))
    w, h = background_image.size
    transparent_background = np.array(background_image)
    transparent_background[:, :, -1] = 128
    transparent_background = Image.fromarray(transparent_background)

    # Create a transparent layer with the same size as the background image
    transparent_layer = np.zeros((h, w, 4))
    for splited_track in splited_tracks:
        if len(splited_track) > 1:
            splited_track = interpolate_trajectory(splited_track, video_length)
            for i in range(len(splited_track)-1):
                start_point = (int(splited_track[i][0]), int(splited_track[i][1]))
                end_point = (int(splited_track[i+1][0]), int(splited_track[i+1][1]))
                vx = end_point[0] - start_point[0]
                vy = end_point[1] - start_point[1]
                arrow_length = np.sqrt(vx**2 + vy**2) + 1e-6
                if i == len(splited_track)-2:
                    cv2.arrowedLine(transparent_layer, start_point, end_point, (255, 0, 0, 192), 2, tipLength=8 / arrow_length)
                else:
                    cv2.line(transparent_layer, start_point, end_point, (255, 0, 0, 192), 2)
        else:
            cv2.circle(transparent_layer, (int(splited_track[0][0]), int(splited_track[0][1])), 2, (255, 0, 0, 192), -1)

    transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
    trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)

    save_dir = os.path.dirname(save_path)
    save_name = os.path.basename(save_path).split('.')[0]
    vis_track = Visualizer(save_dir=save_dir,
                        pad_value=0, 
                        linewidth=10, 
                        mode='optical_flow', 
                        tracks_leave_trace=-1)
    video = np.repeat(np.array(background_image.convert('RGB'))[None, None, ...], splited_tracks.shape[1], 1)
    video = video.transpose(0, 1, 4, 2, 3)
    tracks = splited_tracks[:, :, :2][None].transpose(0, 2, 1, 3)
    depths = splited_tracks[:, :, 2][None].transpose(0, 2, 1)
    video_tracks = vis_track.visualize(torch.from_numpy(video),
                                    torch.from_numpy(tracks),
                                    depths=torch.from_numpy(depths),
                                    filename=save_name, 
                                    is_depth_norm=True,
                                    query_frame=0)

    return trajectory_map, transparent_layer, video_tracks


class Net(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        obj_encoder: MotionEncoder,
        cam_encoder: MotionEncoder,
        fusion_module: FusionModule,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.obj_encoder = obj_encoder
        self.cam_encoder = cam_encoder
        self.fusion_module = fusion_module


class Model():
    def __init__(self, config_path):
        self.config_path = config_path
        self.load_config()
        self.init_model()
        self.init_savedir()
        
        self.output_dir = './tmp'
        os.makedirs(self.output_dir, exist_ok=True)

    def load_config(self):
        self.config = OmegaConf.load(self.config_path)

    def init_model(self):

        if self.config.weight_dtype == "fp16":
            weight_dtype = torch.float16
        else:
            weight_dtype = torch.float32

        vae = AutoencoderKL.from_pretrained(
            self.config.vae_model_path,
        ).to("cuda", dtype=weight_dtype)

        reference_unet = UNet2DConditionModel.from_pretrained(
            self.config.base_model_path,
            subfolder="unet",
        ).to(dtype=weight_dtype, device="cuda")

        inference_config_path = self.config.inference_config
        infer_config = OmegaConf.load(inference_config_path)
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            self.config.base_model_path,
            self.config.mm_path,
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(dtype=weight_dtype, device="cuda")

        obj_encoder = MotionEncoder(
            conditioning_embedding_channels=320, 
            conditioning_channels = 3,
            block_out_channels=(16, 32, 96, 256)
        ).to(device="cuda")

        cam_encoder = MotionEncoder(
            conditioning_embedding_channels=320, 
            conditioning_channels = 1,
            block_out_channels=(16, 32, 96, 256)
        ).to(device="cuda")

        fusion_module = FusionModule(
            fusion_type=self.config.fusion_type,
        ).to(device="cuda")
        
        image_enc = CLIPVisionModelWithProjection.from_pretrained(
            self.config.image_encoder_path
        ).to(dtype=weight_dtype, device="cuda")

        sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        scheduler = DDIMScheduler(**sched_kwargs)

        self.generator = torch.manual_seed(self.config.seed)

        denoising_unet.load_state_dict(
            torch.load(self.config.denoising_unet_path, map_location="cpu"),
            strict=False,
        )
        reference_unet.load_state_dict(
            torch.load(self.config.reference_unet_path, map_location="cpu"),
        )
        obj_encoder.load_state_dict(
            torch.load(self.config.obj_encoder_path, map_location="cpu"),
        )
        cam_encoder.load_state_dict(
            torch.load(self.config.cam_encoder_path, map_location="cpu"),
        )

        pipe = Motion2VideoPipeline(
            vae=vae,
            image_encoder=image_enc,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            obj_encoder=obj_encoder,
            cam_encoder=cam_encoder,
            fusion_module=fusion_module,
            scheduler=scheduler,
        )
        self.pipe = pipe.to("cuda", dtype=weight_dtype)

    def init_savedir(self):
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M")
        save_dir_name = f"{time_str}"

        if 'save_dir' not in self.config:
            self.save_dir = Path(f"output/{date_str}/{save_dir_name}")
        else:
            self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def get_control_objs(self, track_path, ori_sample_size, sample_size, video_length=16, imagepath=None, vo_path=None, cam_type='generated'):
        imagename = imagepath.split('/')[-1].split('.')[0]

        vis = Visualizer(save_dir='./tmp', 
                        grayscale=False, 
                        mode='rainbow_all',
                        pad_value=0, 
                        linewidth=1,
                        tracks_leave_trace=1)
        
        with open(track_path, 'r') as f:
            track_points_user = json.load(f)
            track_points = []
            for splited_track in track_points_user:
                splited_track = np.array(splited_track)
                if splited_track.shape[-1] == 3:
                    splited_track = interpolate_trajectory_3d(splited_track, video_length)
                else:
                    splited_track = interpolate_trajectory(splited_track, video_length)
                splited_track = np.array(splited_track)
                track_points.append(splited_track)
            track_points = np.array(track_points)
        
        if track_points.shape[0] == 0 and not self.config.is_adapted:
            return None, None, None, None

        query_path = os.path.join("./tmp/tmp_query.npy")
        if track_points.shape[0] != 0:
            query_points = track_points.transpose(1, 0, 2)
            np.save(query_path, query_points)
            tracks = query_points

        # get adapted tracks
        if track_points.shape[0] == 0:
            query_path = 'none'
        
        projected_3d_points, points_depth, cam_path = cam_adaptation(imagepath, vo_path, query_path, cam_type, video_length, self.output_dir)
        
        if self.config.is_adapted or track_points.shape[0] == 0:
            tracks = projected_3d_points[:, :, :2]
            tracks = np.concatenate([tracks, np.ones_like(tracks[:, :, 0])[..., None] * 0.5], -1)

        # get depth
        if self.config.is_depth:
            tracks = np.concatenate([tracks.astype(float)[..., :2], points_depth], -1)

        T, _, _ = tracks.shape
        
        tracks[:, :, 0] /= ori_sample_size[1]
        tracks[:, :, 1] /= ori_sample_size[0]
        
        tracks[:, :, 0] *= sample_size[1]
        tracks[:, :, 1] *= sample_size[0]

        tracks[..., 0] = np.clip(tracks[:, :, 0], 0, sample_size[1] - 1)
        tracks[..., 1] = np.clip(tracks[:, :, 1], 0, sample_size[0] - 1)

        tracks = tracks[np.newaxis, :, :, :]
        tracks = torch.tensor(tracks)

        pred_tracks = tracks[:, :, :, :3]

        # vis tracks
        splited_tracks = []
        for i in range(pred_tracks.shape[2]):
            splited_tracks.append(pred_tracks[0, :, i, :3])
        splited_tracks = np.array(splited_tracks)

        video = torch.zeros(T, 3, sample_size[0], sample_size[1])[None].float()

        vis_objs = vis.visualize(video=video, 
                                tracks=pred_tracks[..., :2],
                                filename=Path(track_path).stem,
                                depths=pred_tracks[..., 2],
                                circle_scale=self.config.circle_scale,
                                is_blur=False,
                                is_depth_norm=True,
                                save_video=False
                                )
        tracks = tracks.squeeze().numpy()
        vis_objs = vis_objs.squeeze().numpy()
        guide_value_objs = vis_objs

        return guide_value_objs, splited_tracks, cam_path

    def get_control_cams(self, cam_path, sample_size):
        vr = VideoReader(cam_path)
        cams = vr.get_batch(list(range(0, len(vr)))).asnumpy()[:, :, :, ::-1]
        resized_cams = []
        resized_rgb_cams = []
        for i in range(cams.shape[0]):
            if i == 0:
                cam_height, cam_width = cams[i].shape[:2]
            frame = np.array(Image.fromarray(cams[i]).convert('L').resize([sample_size[1], sample_size[0]]))
            resized_cams.append(frame)
            frame_rgb = np.array(Image.fromarray(cams[i]).convert('RGB').resize([sample_size[1], sample_size[0]]))
            resized_rgb_cams.append(frame_rgb)
        guide_value_cams = np.array(resized_cams)[..., None]
        del vr
        guide_value_cams = guide_value_cams.transpose(0, 3, 1, 2)
        return guide_value_cams
        

    def run(self, ref_image_path, cam_path, track_path, seed=None):
        if not seed:
            seed = self.config.seed
        self.generator = torch.manual_seed(seed)
        video_length = self.config.sample_n_frames
    
        if os.path.exists(cam_path) and cam_path.endswith('.txt'):
            vo_path = cam_path
            cam_type = 'user-provided'
        else:
            vo_path = cam_path
            cam_type = 'generated'
    
        ref_name = Path(ref_image_path).stem

        ref_image_pil = Image.open(ref_image_path).convert("RGB")
        ref_image_path = f'{self.save_dir}/{ref_name}_ref.png'
        w, h = ref_image_pil.size
        ori_width, ori_height = w // 2 * 2, h // 2 * 2
        ref_image_pil.resize((ori_width, ori_height)).save(ref_image_path)

        image_transform = transforms.Compose(
            [transforms.Resize((self.config.H, self.config.W)), transforms.ToTensor()]
        )

        ref_image_tensor = image_transform(ref_image_pil)  # (c, h, w)
        ref_image_pil = Image.fromarray((ref_image_tensor * 255).permute(1, 2, 0).numpy().astype(np.uint8)).convert("RGB")
        
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
        ref_image_tensor = repeat(ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=video_length)
            
        control_objs, splited_tracks, cam_path = self.get_control_objs(
            track_path, 
            ori_sample_size=(ori_height, ori_width), 
            sample_size=(self.config.H, self.config.W), 
            video_length=video_length, 
            imagepath=ref_image_path, 
            vo_path=vo_path.replace('.mp4', '.txt', 1),
            cam_type=cam_type,
            )
        
        control_cams = self.get_control_cams(cam_path, (self.config.H, self.config.W))

        ref_image_path = f'{self.save_dir}/{ref_name}_ref.png'
        ref_image_pil.save(ref_image_path)

        if control_objs is None or self.config.cam_only:
            control_objs = np.zeros_like(control_cams).repeat(3, 1)
        if self.config.obj_only:
            control_cams = np.zeros_like(control_cams)

        control_objs = torch.from_numpy(control_objs).unsqueeze(0).float()
        control_objs = control_objs.transpose(
            1, 2
        )  # (bs, c, f, H, W)
        control_cams = torch.from_numpy(control_cams).unsqueeze(0).float()
        control_cams = control_cams.transpose(
            1, 2
        )  # (bs, c, f, H, W)

        video = self.pipe(
            reference_image=ref_image_pil,
            control_objs=control_objs,
            control_cams=control_cams,
            width=self.config.W,
            height=self.config.H,
            video_length=video_length,
            num_inference_steps=self.config.steps,
            guidance_scale=self.config.guidance_scale,
            generator=self.generator,
            is_obj=self.config.is_obj,
            is_cam=self.config.is_cam
        ).videos

        cam_pattern_postfix = vo_path.split('/')[-1].split('.')[0]
        video_path = f"{self.save_dir}/{ref_name}_{cam_pattern_postfix}_gen.mp4"
        hint_path = f'{self.save_dir}/{ref_name}_{cam_pattern_postfix}_hint.mp4'
        vis_obj_path = f'{self.save_dir}/{ref_name}_{cam_pattern_postfix}_obj.mp4'
        vis_cam_path = f'{self.save_dir}/{ref_name}_{cam_pattern_postfix}_cam.mp4'

        _, _, _ = visualize_tracks(ref_image_path, splited_tracks, video_length, self.config.W, self.config.H, save_path=hint_path)

        torchvision.io.write_video(vis_obj_path, control_objs[0].permute(1, 2, 3, 0).numpy(), fps=8, video_codec='h264', options={'crf': '10'})
        torchvision.io.write_video(vis_cam_path, control_cams[0].permute(1, 2, 3, 0).numpy().repeat(3, 3), fps=8, video_codec='h264', options={'crf': '10'})

        save_videos_grid(
            video,
            video_path,
            n_rows=1,
            fps=8,
        )
    
        return video_path, hint_path, vis_obj_path, vis_cam_path


if __name__ == '__main__':

    config_path = 'configs/eval.yaml'
    config = OmegaConf.load(config_path)
    Model = Model(config_path)

    path_sets = []
    for test_case in config["test_cases"]:
        ref_image_path = list(test_case.keys())[0]
        track_path = test_case[ref_image_path][0]
        cam_path = test_case[ref_image_path][1]
        seed = test_case[ref_image_path][2]
        path_set = {'ref_image_path': ref_image_path,
                    'track_path': track_path,
                    'cam_path': cam_path,
                    'seed': seed,
                    }
        path_sets.append(path_set)

    for path_set in path_sets:
        ref_image_path = path_set['ref_image_path']
        track_path = path_set['track_path']
        cam_path = path_set['cam_path']
        seed = path_set['seed']

        Model.run(ref_image_path, cam_path, track_path, seed)