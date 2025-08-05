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

import safetensors
from einops import repeat
from omegaconf import OmegaConf
from decord import VideoReader
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPVisionModelWithProjection

from src.models.motion_encoder import MotionEncoder
from src.models.fusion_module import FusionModule
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_motion2vid_merge_infer import Motion2VideoPipeline
from src.utils.util import save_videos_grid
from src.utils.utils import interpolate_trajectory, interpolate_trajectory_3d
from src.utils.visualizer import Visualizer

from cam_utils import cam_adaptation


def visualize_tracks(background_image_path, splited_tracks, width, height, save_path='./tmp/hint.mp4'):
    
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
            splited_track = interpolate_trajectory(splited_track, 16)
            splited_track = splited_track[:16]
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

        # load pretrained widths
        if 'load_from_pretrained' in self.config and self.config.load_from_pretrained:
                
            net = Net(
                reference_unet,
                denoising_unet,
                obj_encoder,
                cam_encoder,
                fusion_module,
            )

            checkpoint = safetensors.torch.load_file(self.config.load_from_pretrained)
            net.load_state_dict(checkpoint, strict=False)
            denoising_unet = net.denoising_unet
            reference_unet = net.reference_unet
            obj_encoder = net.obj_encoder
            cam_encoder = net.cam_encoder
            
        else:
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
        
    def get_ref_image(self, path, pick_idx):
        vr = VideoReader(path)
        num_frame = len(vr)
        frame = vr.get_batch([pick_idx]).asnumpy()[0]

        return frame, num_frame

    def get_control_objs(self, track_path, ori_sample_size, sample_size, padding=(0, 0), video_length=16, imagepath=None, vo_path=None, cam_type='generated'):
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
        # os.system(f'python cam_utils.py --T {video_length} --type {cam_type} --imagepath {imagepath} --vo_path {vo_path} --output_dir {self.output_dir} --query_path {query_path}')
        # projected_3d_points = np.load("./tmp/rendered_tracks.npy")
        # cam_path = "./tmp/rendered_cam.mp4"
        
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

        tracks[:, :, 0] += padding[1]
        tracks[:, :, 1] += padding[0]

        tracks[..., 0] = np.clip(tracks[:, :, 0], 0, sample_size[1] - 1)
        tracks[..., 1] = np.clip(tracks[:, :, 1], 0, sample_size[0] - 1)

        tracks = tracks[np.newaxis, :, :, :]
        tracks = torch.tensor(tracks)

        pred_tracks = tracks[:, :, :, :3]

        # vis tracks
        splited_tracks = []
        query_points = []
        for i in range(pred_tracks.shape[2]):
            splited_tracks.append(pred_tracks[0, :, i, :3])
            query_points.append(pred_tracks[0, :, i, :3][0])
        splited_tracks = np.array(splited_tracks)
        query_points = np.array(query_points)

        video = torch.zeros(T, 3, sample_size[0] + 2 * padding[0], sample_size[1] + 2 * padding[1])[None].float()

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

        return guide_value_objs, splited_tracks, query_points, cam_path

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
        
    def run(self, image_name, selected_cam_index, uploaded_cam_file, is_adapted_input=False, is_depth_input=False, video_length=16, seed=12580):
        self.generator = torch.manual_seed(seed)
        print('video_length: ', video_length)
        print('seed: ', seed)
        print('selected_cam_index: ', selected_cam_index)
        print('uploaded_cam_file: ', uploaded_cam_file)

        self.config.is_adapted = bool(is_adapted_input)
        self.config.is_depth = bool(is_depth_input)

        ref_image_path = os.path.join(self.output_dir, f"{image_name}.jpg")
        track_path = os.path.join(self.output_dir, f"{image_name}.json")
    
        if selected_cam_index not in list(range(0, len(cam_cond_patterns))):
            vo_path = uploaded_cam_file
            cam_type = 'user-provided'
        else:
            vo_path = cam_cond_patterns[selected_cam_index]
            cam_type = 'generated'
    
        ref_name = Path(ref_image_path).stem

        ref_image_pil = Image.open(ref_image_path).convert("RGB")
        ref_image_path = f'{self.save_dir}/{ref_name}_ref.png'
        ref_image_pil.save(ref_image_path)

        # already pad
        image_transform = transforms.Compose(
            [transforms.Resize((self.config.H, self.config.W)), transforms.ToTensor()]
        )

        ref_image_tensor = image_transform(ref_image_pil)  # (c, h, w)
        ref_image_pil = Image.fromarray((ref_image_tensor * 255).permute(1, 2, 0).numpy().astype(np.uint8)).convert("RGB")
        
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
        ref_image_tensor = repeat(ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=video_length)
            
        control_objs, splited_tracks, _, cam_path = self.get_control_objs(
            track_path, 
            ori_sample_size=(self.config.H, self.config.W), 
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
        print(control_objs.shape, control_cams.shape)

        if control_objs.shape[2] == control_cams.shape[2]:
            batch_index_cond_obj = np.linspace(0, control_objs.shape[2] - 1, video_length, dtype=int)
            control_objs = control_objs[:, :, batch_index_cond_obj, :, :]
            control_cams = control_cams[:, :, batch_index_cond_obj, :, :]
        else:
            batch_index_cond_obj = np.linspace(0, control_objs.shape[2] - 1, video_length, dtype=int)
            batch_index_cond_cam = np.linspace(0, control_cams.shape[2] - 1, video_length, dtype=int)
            control_objs = control_objs[:, :, batch_index_cond_obj, :, :]
            control_cams = control_cams[:, :, batch_index_cond_cam, :, :]
        print(control_objs.shape, control_cams.shape)

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

        # TODO:
        cam_pattern_postfix = vo_path.split('/')[-1].split('.')[0]
        video_path = f"{self.save_dir}/{ref_name}_{cam_pattern_postfix}_{seed}_gen.mp4"
        hint_path = f'{self.save_dir}/{ref_name}_{cam_pattern_postfix}_{seed}_hint.mp4'
        vis_obj_path = f'{self.save_dir}/{ref_name}_{cam_pattern_postfix}_{seed}_obj.mp4'
        vis_cam_path = f'{self.save_dir}/{ref_name}_{cam_pattern_postfix}_{seed}_cam.mp4'

        if splited_tracks is None:
            save_videos_grid(
                ref_image_tensor,
                hint_path,
                n_rows=2,
                fps=8,
            )
        else:
            print(splited_tracks.shape, f"{self.save_dir}/{ref_name}_{cam_pattern_postfix}_{seed}_trajs.npy")
            np.save(f"{self.save_dir}/{ref_name}_{cam_pattern_postfix}_{seed}_trajs.npy", splited_tracks)
            # vis tracks
            _, _, _ = visualize_tracks(ref_image_path, splited_tracks[:, batch_index_cond_obj, :], self.config.W, self.config.H, save_path=hint_path)

        torchvision.io.write_video(vis_obj_path, control_objs[0].permute(1, 2, 3, 0).numpy(), fps=8, video_codec='h264', options={'crf': '10'})
        torchvision.io.write_video(vis_cam_path, control_cams[0].permute(1, 2, 3, 0).numpy().repeat(3, 3), fps=8, video_codec='h264', options={'crf': '10'})

        save_videos_grid(
            video,
            video_path,
            n_rows=1,
            fps=8,
        )
    
        return video_path, hint_path, vis_obj_path, vis_cam_path

cam_cond_patterns = [
    "./cam_patterns/Still.mp4",
    "./cam_patterns/DollyIn.mp4",
    "./cam_patterns/DollyOut.mp4",
    "./cam_patterns/PanLeft.mp4",
    "./cam_patterns/PanRight.mp4",
    "./cam_patterns/TiltUp.mp4",
    "./cam_patterns/TiltDown.mp4",
    "./cam_patterns/TruckLeft.mp4",
    "./cam_patterns/TruckRight.mp4",
    "./cam_patterns/PedestalUp.mp4",
    "./cam_patterns/PedestalDown.mp4",
    "./cam_patterns/OrbitLeft.mp4",
    "./cam_patterns/OrbitRight.mp4",
    "./cam_patterns/OrbitUp.mp4",
    "./cam_patterns/OrbitDown.mp4",
    "./cam_patterns/ClockwiseRoll.mp4",
    "./cam_patterns/AnticlockwiseRoll.mp4",
]

# cam_pattern_dir = './cam_patterns'
# cam_cond_patterns = [os.path.join(cam_pattern_dir, p) for p in os.listdir(cam_pattern_dir) if p.endswith('.mp4')]

def calculate_centroid(points):
    return np.mean(points, axis=0)

def scale_trajectory(points, original_centroid, scale):
    # Translate points so that original centroid is at origin
    translated_points = points - original_centroid
    
    # Scale the translated points
    scaled_translated_points = translated_points * scale
    
    return scaled_translated_points

def translate_to_start_point(points, start_point):
    # Calculate the translation vector needed to move the first point to the start point
    translation_vector = start_point - points[0]
    
    # Apply the translation vector to all points
    translated_points = points + translation_vector
    
    return translated_points


def resize_with_padding(img, expected_size):
    from PIL import Image, ImageOps
    x, y = img.size
    w, h = expected_size[1], expected_size[0]

    if x / y < w / h:  # resize by h, pad by w 
        new_h, new_w = h, int(h / y * x) // 2 * 2
        pad_h, pad_w = 0, (w - new_w) // 2
    else:
        new_h, new_w = int(w / x * y) // 2 * 2, w
        pad_h, pad_w = (h - new_h) // 2, 0

    padding = (pad_w, pad_h, pad_w, pad_h)

    # Padding the image
    color = (0, 0, 0)  # White padding, change this tuple to change the padding color
    new_img = ImageOps.expand(img, padding, fill=color).resize([w, h])
    return new_img


def resize_and_crop(img, expected_size):
    original_width, original_height = img.size
    expected_height, expected_width = expected_size

    # Calculate the scaling factor for width and height
    scale_w = expected_width / original_width
    scale_h = expected_height / original_height

    # Choose the smaller scale factor to ensure the entire image fits within the expected size
    scale_factor = max(scale_w, scale_h)

    # Calculate new dimensions based on the chosen scale factor
    new_width = int(original_width * scale_factor // 2 * 2)
    new_height = int(original_height * scale_factor // 2 * 2)

    # Resize the image to these new dimensions
    img_resized = img.resize((new_width, new_height))

    # Calculate cropping area to center the image after resizing
    left = (new_width - expected_width) / 2
    top = (new_height - expected_height) / 2
    right = (new_width + expected_width) / 2
    bottom = (new_height + expected_height) / 2

    # Ensure that the calculated coordinates are non-negative integers
    left, top, right, bottom = map(int, [max(left, 0), max(top, 0), min(right, new_width), min(bottom, new_height)])

    # Crop the image to the final size
    img_cropped = img_resized.crop((left, top, right, bottom))

    return img_cropped


with gr.Blocks() as demo:
    gr.Markdown("""<h1 align="center">Perception-as-Control</h1><br>""")
    
    config_path = './configs/gradio/motion3d_gradio.yaml'
    Model = Model(config_path)
    input_image_path = gr.State()
    correspond_frame_path = gr.State()
    tracking_points = gr.State([])
    driven_points = gr.State([])
    depth_value = gr.State(0.5)
    image_name = gr.State()
    selected_cam_index = gr.State()
    uploaded_cam_file = gr.State()

    output_dir = './tmp'

    def preprocess_image(image, sample_size, fit_type='crop'):
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image).convert('RGB')
            image_name = 'current'
        else:
            image_pil = Image.open(image.name).convert('RGB')
            image_name = image.name.split('/')[-1].split('.')[0]

        if fit_type == 'crop':
            image_pil = resize_and_crop(image_pil, sample_size)
        else:
            image_pil = resize_with_padding(image_pil, sample_size)

        raw_w, raw_h = image_pil.size
        print('Resized W: {}, H: {}'.format(raw_w, raw_h))

        input_image_path = os.path.join(output_dir, image_name + '.jpg')
        image_pil.save(input_image_path)

        return input_image_path, input_image_path, gr.State([]), image_name, input_image_path

    def video_to_numpy_array(video_path):
        video = cv2.VideoCapture(video_path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for i in range(num_frames):
            ret, frame = video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames.append(frame)
            else:
                break
        video.release()
        frames = np.stack(frames, axis=0)
        return frames

    def convert_video_to_mp4(input_audio_file, output_wav_file):
        if input_audio_file.endswith('.mp4'):
            return
        video_np = np.uint8(video_to_numpy_array(input_audio_file))
        torchvision.io.write_video(
            output_wav_file, 
            video_np, 
            fps=25, video_codec='h264', options={'crf': '10'}
        )
        return

    def preprocess_video(driven_video, image_name):
        frames = video_to_numpy_array(driven_video)
        first_frame = Image.fromarray(frames[0]).convert('RGB')
        correspond_frame_path = os.path.join(output_dir, image_name + '_driven.jpg')
        first_frame.save(correspond_frame_path)

        return correspond_frame_path, correspond_frame_path, gr.State([]), driven_video

    def add_traj(tracking_points):
        if len(tracking_points.constructor_args['value']) != 0 and tracking_points.constructor_args['value'][-1] == []:
            return tracking_points
        tracking_points.constructor_args['value'].append([])
        return tracking_points
    
    def confirm_traj(tracking_points, image_name):
        print(tracking_points.constructor_args['value'])

        import json

        json_data = json.dumps(tracking_points.constructor_args['value'])
        file_path = os.path.join(output_dir, image_name + '.json')
        with open(file_path, 'w') as file:
            file.write(json_data)
        print('save trajs in {}'.format(file_path))

        query_points = []
        for track in tracking_points.constructor_args['value']:
            query_points.append(track[0][:2])
        query_path = os.path.join(output_dir, image_name + '_query.npy')
        np.save(query_path, query_points)
        print('save query points in {}'.format(file_path))

        return tracking_points
    
    def confirm_driven(driven_points, image_name):
        print(driven_points.constructor_args['value'])

        query_path = os.path.join(output_dir, image_name + '_corres_query.npy')
        query_points = driven_points.constructor_args['value']
        print(np.array(query_points).shape)
        np.save(query_path, query_points)
        
        return driven_points

    def delete_tracking_points(tracking_points, input_image_path):
        
        if len(tracking_points.constructor_args['value']) > 0:
            tracking_points.constructor_args['value'].pop()

        transparent_background = Image.open(input_image_path).convert('RGBA')
        w, h = transparent_background.size
        transparent_layer = np.zeros((h, w, 4))
        for track in tracking_points.constructor_args['value']:
            if len(track) > 1:
                for i in range(len(track)-1):
                    start_point = track[i][:2]
                    end_point = track[i+1][:2]
                    vx = end_point[0] - start_point[0]
                    vy = end_point[1] - start_point[1]
                    arrow_length = np.sqrt(vx**2 + vy**2)
                    if i == len(track)-2:
                        cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=8 / arrow_length)
                    else:
                        cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
            else:
                cv2.circle(transparent_layer, tuple(track[0][:2]), 5, (255, 0, 0, 255), -1)

        transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
        trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)

        return tracking_points, trajectory_map


    def add_tracking_points(tracking_points, input_image_path, image_name, depth_value, evt: gr.SelectData):  # SelectData is a subclass of EventData
        print(f"You selected {evt.value} at {evt.index} from {evt.target}")
        if len(tracking_points.constructor_args['value']) == 0:
            tracking_points.constructor_args['value'].append([])
        
        tracking_points.constructor_args['value'][-1].append(evt.index + [depth_value])

        transparent_background = Image.open(input_image_path).convert('RGBA')
        w, h = transparent_background.size
        transparent_layer = np.zeros((h, w, 4))
        for track in tracking_points.constructor_args['value']:
            if len(track) > 1:
                for i in range(len(track)-1):
                    start_point = track[i][:2]
                    end_point = track[i+1][:2]
                    vx = end_point[0] - start_point[0]
                    vy = end_point[1] - start_point[1]
                    arrow_length = np.sqrt(vx**2 + vy**2) + 1e-6
                    if i == len(track)-2:
                        cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=8 / arrow_length)
                    else:
                        cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
            else:
                cv2.circle(transparent_layer, tuple(track[0][:2]), 5, (255, 0, 0, 255), -1)

        transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
        trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
        trajectory_map.save(os.path.join(output_dir, image_name + '_hint.png'))

        return tracking_points, trajectory_map
    

    def dup_tracking_points(tracking_points, input_image_path, image_name, traj_scale):

        root = tracking_points.constructor_args['value'][-1][0]
        last_traj = tracking_points.constructor_args['value'][-2]
        last_root = tracking_points.constructor_args['value'][-2][0]

        last_traj = np.array(last_traj)[:, :2]
        last_root = np.array(last_root)[:2]
        root = np.array(root)[:2]

        # Calculate original centroid
        original_centroid = calculate_centroid(last_traj)
        # Get scaled trajectory with new centroid
        scaled_trajectory = scale_trajectory(last_traj, original_centroid, traj_scale)
        dup_traj = translate_to_start_point(scaled_trajectory, root)
        dup_traj = dup_traj.astype(int).tolist()

        dup_traj_depth = [p + [0.5] for p in dup_traj]
        
        tracking_points.constructor_args['value'].pop()
        tracking_points.constructor_args['value'].append(dup_traj_depth)

        transparent_background = Image.open(input_image_path).convert('RGBA')
        w, h = transparent_background.size
        transparent_layer = np.zeros((h, w, 4))
        for track in tracking_points.constructor_args['value']:
            if len(track) > 1:
                for i in range(len(track)-1):
                    start_point = track[i][:2]
                    end_point = track[i+1][:2]
                    vx = end_point[0] - start_point[0]
                    vy = end_point[1] - start_point[1]
                    arrow_length = np.sqrt(vx**2 + vy**2) + 1e-6
                    if i == len(track)-2:
                        cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=8 / arrow_length)
                    else:
                        cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
            else:
                cv2.circle(transparent_layer, tuple(track[0][:2]), 5, (255, 0, 0, 255), -1)

        transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
        trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
        trajectory_map.save(os.path.join(output_dir, image_name + '_hint.png'))

        return tracking_points, trajectory_map

    
    def add_driven_points(driven_points, correspond_frame_path, image_name, evt: gr.SelectData):  # SelectData is a subclass of EventData
        print(f"You selected {evt.value} at {evt.index} from {evt.target}")

        driven_points.constructor_args['value'].append(evt.index)

        transparent_background = Image.open(correspond_frame_path).convert('RGBA')
        w, h = transparent_background.size
        transparent_layer = np.zeros((h, w, 4))
        for point in driven_points.constructor_args['value']:
            cv2.circle(transparent_layer, tuple(point), 10, (0, 255, 0, 255), -1)

        transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
        trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
        trajectory_map.save(os.path.join(output_dir, image_name + '_driven.png'))

        return tracking_points, trajectory_map
    
    def show_selected_video(selected_state: gr.SelectData):
        return cam_cond_patterns[selected_state.index], selected_state.index

    with gr.Accordion(label="🧭 User Guide:", open=True, elem_id="accordion"):
        with gr.Row(equal_height=True):
            gr.Markdown("""
            - ⭐️ <b>Step1: </b>Click "Upload Image" and upload an image from local folder. 
            - ⭐️ <b>Step2: </b>Click "Add Trajectory" and draw trajectories on the uploaded image.
            - ⭐️ <b>Step3: </b>Confirm the current trajectories if finished. 
            - ⭐️ <b>Step4: </b>Select a camera movement pattern or upload your camera file. 
            - ⭐️ <b>Step5: </b>Set video length and seed if you want. 
            - ⭐️ <b>Step6: </b>Click "Run" to generate a video according to those conditions.
            - ⭐️ <b>Step7 (optinal): </b>The generated results contain randomness. If you're not satisfied, change the seed and click "Run" to try again.
            """)

    with gr.Row():
        with gr.Column(scale=3):
            image_upload_button = gr.UploadButton(label="Upload Image",file_types=["image"])
            
            def get_selection(selection):
                return selection
            fit_type = gr.State('crop')
            choice_dropdown = gr.Dropdown(
                choices=['crop', 'pad'],
                label="Please choose how to fit the input image",
                value='crop'
            )
            choice_dropdown.change(fn=get_selection, inputs=[choice_dropdown], outputs=fit_type)

            image_upload = gr.Image(label="Upload Driven Image", interactive=True)

        with gr.Column(scale=3):
            add_traj_button = gr.Button(value="Add Trajectory")
            gr.Markdown("""
                        - ⭐️ <b> Click "Add Trajectory" every time before adding a new one. 
                        - ⭐️ <b> Not required for camera-only motion control (automatically set).
                        - ⭐️ <b> Please set some static ones for collaborative motion control.
                        """)

            delete_traj_button = gr.Button(value="Delete Last Trajectory")
            
            gr.Markdown("""⭐️ <b> Draw the start point and click "Duplicate Last Trajectory" for a duplicated one.""")
            dup_last_traj_button = gr.Button(value="Duplicate Last Trajectory")
            traj_scale = gr.Slider(label='Trajectory Scale for Duplicate', 
                                             minimum=0, 
                                             maximum=2., 
                                             step=0.01, 
                                             value=1.0)
    with gr.Row():
        with gr.Column(scale=3):

            gr.Markdown("""- ⭐️ <b> Please draw trajectories here.""")
            input_image = gr.Image(label="Draw Trajectory Here", interactive=True)

            gr.Markdown("""- ⭐️ <b> Please confirm before going to the next step.""")
            confirm_traj_button = gr.Button(value="Confirm Current Trajectories")

    with gr.Row():
        def checkbox_selected(is_checked):
            return is_checked
        
        with gr.Column(scale=3):
            gr.Markdown("""- ⭐️ <b> Please select a camera movement pattern on the right.""")
            cam_cond = gr.Video(label="Selected Camera Movement")

            checkbox_adapted = gr.Checkbox(label="Trajectories adaptation based on camera movement", value=False)
            is_adapted_input = gr.State(True)
            checkbox_adapted.change(fn=checkbox_selected, inputs=[checkbox_adapted], outputs=is_adapted_input)

            checkbox_depth = gr.Checkbox(label="Automatic depth estimation for trajectory starting points", value=False)
            is_depth_input = gr.State(True)
            checkbox_depth.change(fn=checkbox_selected, inputs=[checkbox_depth], outputs=is_depth_input)

        with gr.Column(scale=3):

            cam_gallery = gr.Gallery(label="Camera Movement Patterns", columns=8, height=320,
                                  value=[(p, f"{p.split('/')[-1].split('.')[0]}") for p in cam_cond_patterns],
                                  allow_preview=False, 
                                  show_share_button=False,
                                  selected_index=-1,
                                  container=False)
            cam_gallery.select(show_selected_video, inputs=[], outputs=[cam_cond, selected_cam_index])


            def upload_file(file):
                try:
                    poses = np.loadtxt(file)
                    vo_path = './tmp/tmp_vo.txt'
                    np.savetxt(vo_path, poses)
                    return vo_path, poses
                except:
                    return vo_path, "Format not supported."
                
            file_upload_button = gr.UploadButton(label="Upload Camera File", file_types=[".txt"])
            gr.Markdown("""- ⭐️ <b> Require .txt file with T lines, 12 elements represent camera extrinsic parameters for each line, sep by comma.""")
            output_text = gr.Textbox(label="File content", interactive=False)
            file_upload_button.upload(fn=upload_file, inputs=[file_upload_button], outputs=[uploaded_cam_file, output_text])
             
        with gr.Column(scale=3):
            run_button = gr.Button(value="Run")

            def update_video_length(video_length_input):
                if video_length_input is None or video_length_input == "":
                    return 16
                else:
                    number = int(video_length_input)
                    return number
        
            video_length_input = gr.Number(label="Set video length", value=16, step=1)
            video_length = gr.State(16)
            video_length_input.change(
                fn=update_video_length,
                inputs=[video_length_input],
                outputs=[video_length]
            )

            def update_state(seed_input):
                if seed_input is None or seed_input == "":
                    return 12580
                else:
                    number = int(seed_input)
                    return number
        
            seed_input = gr.Number(label="Set a seed if you want", value=12580, step=1)
            seed = gr.State(12580)
            seed_input.change(
                fn=update_state,
                inputs=[seed_input],
                outputs=[seed]
            )

            output_video = gr.Video(label="Generated Video")

    with gr.Row():   
        with gr.Column(scale=1):
            hint_image = gr.Video(label="Trajectory Visualization")
        with gr.Column(scale=1):
            vis_cam = gr.Video(label="Final Camera Movement")
        with gr.Column(scale=1):
            vis_obj = gr.Video(label="Final Object Movement")

    sample_size = gr.State((Model.config.H, Model.config.W))
    image_upload_button.upload(preprocess_image, [image_upload_button, sample_size, fit_type], [input_image, input_image_path, tracking_points, image_name, image_upload])
    image_upload.upload(preprocess_image, [image_upload, sample_size, fit_type], [input_image, input_image_path, tracking_points, image_name, image_upload])

    add_traj_button.click(add_traj, [tracking_points], [tracking_points])
    input_image.select(add_tracking_points, [tracking_points, input_image_path, image_name, depth_value], [tracking_points, input_image])

    dup_last_traj_button.click(dup_tracking_points, [tracking_points, input_image_path, image_name, traj_scale], [tracking_points, input_image])

    delete_traj_button.click(delete_tracking_points, [tracking_points, input_image_path], [tracking_points, input_image])
    confirm_traj_button.click(confirm_traj, [tracking_points, image_name], [tracking_points])

    run_button.click(Model.run, [image_name, selected_cam_index, uploaded_cam_file, is_adapted_input, is_depth_input, video_length, seed], [output_video, hint_image, vis_obj, vis_cam])

    demo.launch(server_name="0.0.0.0", debug=True, server_port=8017)


