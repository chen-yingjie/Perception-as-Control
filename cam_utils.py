import os
import sys
import cv2
import copy
import math
import shutil
import random
import argparse
import subprocess
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

import torch
import numpy as np
# import open3d as o3d
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform, FoVPerspectiveCameras,
                                PointLights, RasterizationSettings, PointsRasterizer, PointsRasterizationSettings,
                                MeshRenderer, MeshRasterizer, SoftPhongShader, PointsRenderer, AlphaCompositor,
                                blending, Textures)
from pytorch3d.io import load_obj

from depth_anything_v2.metric_depth_estimation import depth_estimation


def draw_points(image, points, output_path):
    image = image.copy()
    for point in points:
        u, v = map(int, point[:2])
        image = cv2.circle(image, (u, v), 5, (0, 0, 255), -1)

    return image


class Render():
    def __init__(self, img_size=(320, 576), focal=100, device='cpu', 
        ply_path=None, uvmap_obj_path=None, is_pointcloud=False):
        
        self.img_size = img_size
        self.focal = focal
        self.device = device
        self.is_pointcloud = is_pointcloud

        if self.is_pointcloud:
            if ply_path != None:
                self.model = self.load_point_cloud(ply_path)
        else:
            if uvmap_obj_path != None:
                self.model = self.load_mesh(uvmap_obj_path)

        self.set_renderer()
        self.get_renderer()

    def get_points(self):
        return self.vts.detach().clone().cpu().numpy()

    def load_point_cloud(self, ply_path):
        ply = o3d.io.read_point_cloud(ply_path)
        self.vts = torch.Tensor(np.asarray(ply.points)).to(self.device)
        rgbs = np.asarray(ply.colors)[:, ::-1].copy()
        self.rgbs = torch.Tensor(rgbs).to(self.device) * 255

        colored_pointclouds = pytorch3d.structures.Pointclouds(points=[self.vts], features=[self.rgbs])
        colored_pointclouds.center = torch.tensor([0, 0, 0])
        
        return colored_pointclouds
    
    def load_mesh(self, uvmap_obj_path):
        batch_size = 1

        verts, faces, aux = load_obj(uvmap_obj_path)
        verts_uvs = aux.verts_uvs[None, ...].repeat(batch_size, 1, 1)
        faces_uvs = faces.textures_idx[None, ...].repeat(batch_size, 1, 1)

        tex_maps = aux.texture_images
        texture_image = list(tex_maps.values())[0]
        texture_image = texture_image * 255.0
        texture_maps = torch.tensor(texture_image).unsqueeze(0).repeat(
            batch_size, 1, 1, 1).float()
        tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_maps).to(self.device)
        
        mesh = Meshes(
            verts=torch.tensor(verts[None, ...]).float().to(self.device),
            faces=torch.tensor(faces.verts_idx[None, ...]).float().to(self.device),
            textures=tex
        )

        return mesh

    def update_view(self, distance=1.0, pitch=0.0, azimuth=0.0, roll=0.0, target_point=[0.0, 0.0, 1.0]):
        target_point = torch.tensor(target_point, device=self.device)

        pitch_rad = math.radians(pitch)
        azimuth_rad = math.radians(azimuth)

        z = distance * math.cos(pitch_rad) * math.cos(azimuth_rad)
        x = distance * math.cos(pitch_rad) * math.sin(azimuth_rad)
        y = distance * math.sin(pitch_rad)
        self.camera_position = target_point - torch.tensor([x, y, z], device=self.device)

        R, T = look_at_view_transform(
            eye=self.camera_position.unsqueeze(0),
            at=target_point.unsqueeze(0),
            up=torch.tensor([0.0, 1.0, 0.0], device=self.device).unsqueeze(0),
            device=self.device
        )

        def rotate_roll(R, roll):
            roll_rad = math.radians(roll)
            roll_matrix = torch.tensor([
                [math.cos(roll_rad), -math.sin(roll_rad), 0],
                [math.sin(roll_rad), math.cos(roll_rad), 0],
                [0, 0, 1],
            ], device=self.device)
            return torch.matmul(R, roll_matrix)

        if roll != 0:
            R = rotate_roll(R, roll)

        self.cameras.R = R
        self.cameras.T = T

        return R.cpu().squeeze().numpy(), T.cpu().squeeze().numpy()

    def update_RT(self, RT, S=np.eye(3)):

        if self.is_pointcloud:
            R_norm = torch.tensor([[[1., 0., 0.],
                                    [0., 1., 0.],
                                    [0., 0., 1.]]])
            T_norm = torch.tensor([[0., 0., 0.]])
        else:
            R_norm, T_norm = look_at_view_transform(-50, 0, 0)

        RT_norm = torch.cat((R_norm.squeeze(0), T_norm.squeeze(0)[..., None]),
                            1).numpy()
        RT_norm = np.concatenate((RT_norm, np.array([0, 0, 0, 1])[None, ...]),
                                0)

        R = RT[:, :3]
        T = RT[:, 3]
        R = S @ R @ S
        T = S @ T

        RT = np.concatenate((R, T[..., None]), 1)
        RT = np.concatenate((RT, np.array([0, 0, 0, 1])[None, ...]), 0)

        RT = RT @ RT_norm
        R = RT[:3, :3][None, ...]
        T = RT[:3, 3][None, ...]

        self.cameras.R = torch.tensor(R).to(self.device)
        self.cameras.T = torch.tensor(T).to(self.device)

        return R, T

    def interpolate_slerp(self, RTs, T=16):
        Rs = [RT[0] for RT in RTs]
        Ts = [RT[1] for RT in RTs]

        times = np.linspace(0, 1, len(RTs))
        interp_times = np.linspace(0, 1, T)
        quaternions = [R.from_matrix(R_).as_quat() for R_ in Rs]
        interp_Ts = interp1d(times, Ts, axis=0)(interp_times)

        def slerp(q1, q2, t):
            q1 = R.from_quat(q1)
            q2 = R.from_quat(q2)
            return (q1 * (q2 * q1.inv()) ** t).as_quat()

        interp_quaternions = []
        for i in range(len(quaternions) - 1):
            t = (interp_times - times[i]) / (times[i + 1] - times[i])
            valid_t = t[(t >= 0) & (t <= 1)]
            for t_val in valid_t:
                interp_quaternions.append(slerp(quaternions[i], quaternions[i + 1], t_val).squeeze())

        if len(interp_quaternions) < T:
            interp_quaternions.extend([quaternions[-1]] * (T - len(interp_quaternions)))

        interp_quaternions = np.array(interp_quaternions)
        interp_Rs = [R.from_quat(q).as_matrix() for q in interp_quaternions]
        interpolated_poses = []
        for t in range(T):
            interpolated_poses.append((interp_Rs[t].squeeze(), interp_Ts[t].squeeze()))

        return interpolated_poses

    def set_renderer(self):
        
        self.lights = PointLights(device=self.device,
                            location=[[0.0, 0.0, 1e5]],
                            ambient_color=[[1, 1, 1]],
                            specular_color=[[0., 0., 0.]],
                            diffuse_color=[[0., 0., 0.]])

        if self.is_pointcloud:
            self.raster_settings = PointsRasterizationSettings(
                image_size=self.img_size,
                radius=0.01,
                points_per_pixel=10,
                bin_size=0,
            )
        else:
            self.raster_settings = RasterizationSettings(
                image_size=self.img_size,
                blur_radius=0.0,
                faces_per_pixel=10,
                bin_size=0,
            )
            self.blend_params = blending.BlendParams(background_color=[0, 0, 0])

        R_norm = torch.tensor([[[1., 0., 0.],
                                [0., 1., 0.],
                                [0., 0., 1.]]])
        T_norm = torch.tensor([[0., 0., 0.]])

        self.cameras = FoVPerspectiveCameras(
            device=self.device,
            R=R_norm,
            T=T_norm,
            znear=0.01,
            zfar=200,
            fov=2 * np.arctan(self.img_size[0] // 2 / self.focal) * 180. / np.pi,
            aspect_ratio=1.0,
        )

        return

    def get_renderer(self):
        if self.is_pointcloud:

            self.renderer = PointsRenderer(
                rasterizer=PointsRasterizer(cameras=self.cameras, raster_settings=self.raster_settings),
                compositor=AlphaCompositor()
            )
        else:
            self.renderer = MeshRenderer(rasterizer=MeshRasterizer(
                cameras=self.cameras, raster_settings=self.raster_settings),
                                    shader=SoftPhongShader(device=self.device,
                                                        cameras=self.cameras,
                                                        lights=self.lights,
                                                        blend_params=self.blend_params))

        return

    def render(self):
        rendered_img = self.renderer(self.model).cpu()
        return rendered_img

    def project_vs(self, coords_3d=None):
        if coords_3d is None:
            coords_3d = self.model.points_padded()[0]
        else:
            coords_3d = torch.tensor(coords_3d).to(self.device).float()
       
        coords_3d = self.cameras.transform_points_screen(coords_3d, image_size=self.img_size)

        return coords_3d
    
    def get_vertical_distances(self, points_world):
        R = self.cameras.R.cpu().numpy()[0] 
        T = self.cameras.T.cpu().numpy()[0]      
        T = T[:, np.newaxis]  
        points_camera = (R @ points_world.T + T).T
        vertical_distances = points_camera[:, 2]

        return vertical_distances


def generate_query(img_size, video_length):

    height, width = img_size[0], img_size[1]
    query_points = []
    for i in range(0, width, width // 30):
        for j in range(0, height, height // 30):
            query_points.append([i, j])
    query_points = np.array(query_points)[None, ...]
    query_points = np.repeat(query_points, video_length, 0)
    return query_points


def estimate_depth(imagepath, query_points, img_size, focal, output_dir):
    
    height, width = img_size[0], img_size[1]
    imagename = imagepath.split('/')[-1].split('.')[0]

    metric_depth, relative_depth, _ = depth_estimation(imagepath)

    query_points = np.round(query_points)
    relative_points_depth = [relative_depth[int(p[1]), int(p[0])] if 0 <= int(p[1]) < relative_depth.shape[0] and 0 <= int(p[0]) < relative_depth.shape[1] else 0.5 for p in query_points[0]]
    relative_points_depth = np.array(relative_points_depth)[None, ...]
    relative_points_depth = np.repeat(relative_points_depth, query_points.shape[0], 0)[..., None]

    points_depth = [metric_depth[int(p[1]), int(p[0])] if 0 <= int(p[1]) < metric_depth.shape[0] and 0 <= int(p[0]) < metric_depth.shape[1] else 0.5 for p in query_points[0]]
    points_depth = np.array(points_depth)[None, ...]
    points_depth = np.repeat(points_depth, query_points.shape[0], 0)[..., None]
    points_3d = np.concatenate([query_points, points_depth], -1)

    points_3d[:, :, 0] = -(points_3d[:, :, 0] - width / 2) / focal
    points_3d[:, :, 1] = -(points_3d[:, :, 1] - height / 2) / focal
    
    points_3d[:, :, 0] *= points_3d[:, :, 2]
    points_3d[:, :, 1] *= points_3d[:, :, 2]

    return points_3d, relative_points_depth


def generate_vo(renderer, pattern_name):
    RTs = []
    if pattern_name == 'Still':
        R, T = renderer.update_view(distance=5.0, pitch=0.0, azimuth=0.0, roll=0.0, target_point=[0.0, 0.0, 5.0])
        RTs.append((R, T))
        R, T = renderer.update_view(distance=5.0, pitch=0.0, azimuth=0.0, roll=0.0, target_point=[0.0, 0.0, 5.0])
        RTs.append((R, T))
    elif pattern_name == 'Shaking':
        R, T = renderer.update_view(distance=5.0, pitch=0.0, azimuth=0.0, roll=0.0, target_point=[0.0, 0.0, 5.0])
        RTs.append((R, T))
        for idx in range(0, 8):
            delta_x = random.randint(0, 1) // 3
            pitch = random.randint(0, 1)
            roll = random.randint(0, 1)
            R, T = renderer.update_view(distance=1.0, pitch=pitch, azimuth=0.0, roll=roll, target_point=[delta_x, 0.0, 1.0])
            RTs.append((R, T))
    elif pattern_name == 'OrbitLeft':
        for azimuth in np.linspace(0.0, -15.0, 8):
            R, T = renderer.update_view(distance=3.0, pitch=0.0, azimuth=azimuth, roll=0.0, target_point=[0.0, 0.0, 3.0])
            RTs.append((R, T))
    elif pattern_name == 'OrbitRight':
        for azimuth in np.linspace(0.0, 15.0, 8):
            R, T = renderer.update_view(distance=3.0, pitch=0.0, azimuth=azimuth, roll=0.0, target_point=[0.0, 0.0, 3.0])
            RTs.append((R, T))
    elif pattern_name == 'OrbitUp':
        for pitch in np.linspace(0.0, -15.0, 8):
            R, T = renderer.update_view(distance=3.0, pitch=pitch, azimuth=0.0, roll=0.0, target_point=[0.0, 0.0, 3.0])
            RTs.append((R, T))
    elif pattern_name == 'OrbitDown':
        for pitch in np.linspace(0.0, 15.0, 8):
            R, T = renderer.update_view(distance=3.0, pitch=pitch, azimuth=0.0, roll=0.0, target_point=[0.0, 0.0, 3.0])
            RTs.append((R, T))
    elif pattern_name == 'TruckLeft':
        R, T = renderer.update_view(distance=1.0, pitch=0.0, azimuth=0.0, roll=0.0, target_point=[0.0, 0.0, 1.0])
        RTs.append((R, T))
        R, T = renderer.update_view(distance=1.0, pitch=0.0, azimuth=0.0, roll=0.0, target_point=[0.5, 0.0, 1.0])
        RTs.append((R, T))
    elif pattern_name == 'TruckRight':
        R, T = renderer.update_view(distance=1.0, pitch=0.0, azimuth=0.0, roll=0.0, target_point=[0.0, 0.0, 1.0])
        RTs.append((R, T))
        R, T = renderer.update_view(distance=1.0, pitch=0.0, azimuth=0.0, roll=0.0, target_point=[-0.5, 0.0, 1.0])
        RTs.append((R, T))
    elif pattern_name == 'PedestalUp':
        R, T = renderer.update_view(distance=1.0, pitch=0.0, azimuth=0.0, roll=0.0, target_point=[0.0, 0.0, 1.0])
        RTs.append((R, T))
        R, T = renderer.update_view(distance=1.0, pitch=0.0, azimuth=0.0, roll=0.0, target_point=[0.0, 0.5, 1.0])
        RTs.append((R, T))
    elif pattern_name == 'PedestalDown':
        R, T = renderer.update_view(distance=1.0, pitch=0.0, azimuth=0.0, roll=0.0, target_point=[0.0, 0.0, 1.0])
        RTs.append((R, T))
        R, T = renderer.update_view(distance=1.0, pitch=0.0, azimuth=0.0, roll=0.0, target_point=[0.0, -0.5, 1.0])
        RTs.append((R, T))
    elif pattern_name == 'PanLeft':
        ori_distance = 1.0
        for azimuth in [0.0, 10.0]:
            azimuth_rad = math.radians(azimuth)
            distance = ori_distance * (1.0 / math.cos(azimuth_rad))
            target_point = [ori_distance * math.tanh(azimuth_rad), 0.0, 1.0]
            R, T = renderer.update_view(distance=distance, pitch=0.0, azimuth=azimuth, roll=0.0, target_point=target_point)
            RTs.append((R, T))
    elif pattern_name == 'PanRight':
        ori_distance = 1.0
        for azimuth in [0.0, -10.0]:
            azimuth_rad = math.radians(azimuth)
            distance = ori_distance * (1.0 / math.cos(azimuth_rad))
            target_point = [ori_distance * math.tanh(azimuth_rad), 0.0, 1.0]
            R, T = renderer.update_view(distance=distance, pitch=0.0, azimuth=azimuth, roll=0.0, target_point=target_point)
            RTs.append((R, T))
    elif pattern_name == 'TiltUp':
        ori_distance = 1.0
        for pitch in [0.0, 10.0]:
            pitch_rad = math.radians(pitch)
            distance = ori_distance * (1.0 / math.cos(pitch_rad))
            target_point = [0.0, ori_distance * math.tanh(pitch_rad), 1.0]
            R, T = renderer.update_view(distance=distance, pitch=pitch, azimuth=0.0, roll=0.0, target_point=target_point)
            RTs.append((R, T))
    elif pattern_name == 'TiltDown':
        ori_distance = 1.0
        for pitch in [0.0, -10.0]:
            pitch_rad = math.radians(pitch)
            distance = ori_distance * (1.0 / math.cos(pitch_rad))
            target_point = [0.0, ori_distance * math.tanh(pitch_rad), 1.0]
            R, T = renderer.update_view(distance=distance, pitch=pitch, azimuth=0.0, roll=0.0, target_point=target_point)
            RTs.append((R, T))
    elif pattern_name == 'ClockwiseRoll':
        R, T = renderer.update_view(distance=1.0, pitch=0.0, azimuth=0.0, roll=0.0, target_point=[0.0, 0.0, 1.0])
        RTs.append((R, T))
        R, T = renderer.update_view(distance=1.0, pitch=0.0, azimuth=0.0, roll=20.0, target_point=[0.0, 0.0, 1.0])
        RTs.append((R, T))
    elif pattern_name == 'AnticlockwiseRoll':
        R, T = renderer.update_view(distance=1.0, pitch=0.0, azimuth=0.0, roll=0.0, target_point=[0.0, 0.0, 1.0])
        RTs.append((R, T))
        R, T = renderer.update_view(distance=1.0, pitch=0.0, azimuth=0.0, roll=-20.0, target_point=[0.0, 0.0, 1.0])
        RTs.append((R, T))
    elif pattern_name == 'DollyIn':
        R, T = renderer.update_view(distance=10.0, pitch=0.0, azimuth=0.0, roll=0.0, target_point=[0.0, 0.0, 10.0])
        RTs.append((R, T))
        R, T = renderer.update_view(distance=9.5, pitch=0.0, azimuth=0.0, roll=0.0, target_point=[0.0, 0.0, 10.0])
        RTs.append((R, T))
    elif pattern_name == 'DollyOut':
        R, T = renderer.update_view(distance=1.0, pitch=0.0, azimuth=0.0, roll=0.0, target_point=[0.0, 0.0, 1.0])
        RTs.append((R, T))
        R, T = renderer.update_view(distance=2.0, pitch=0.0, azimuth=0.0, roll=0.0, target_point=[0.0, 0.0, 1.0])
        RTs.append((R, T))

    return RTs


def cam_adaptation(imagepath, vo_path, query_path, cam_type, video_length, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    tmp_folder = './tmp_cam'
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=False)

    img = Image.open(imagepath).convert('RGB')
    img_size = (img.size[1], img.size[0])
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    S = np.eye(3)
    focal_point = 500
    uvmap_obj_path = './mesh/world_envelope.obj'

    renderer = Render(img_size=img_size, device=device, uvmap_obj_path=uvmap_obj_path, is_pointcloud=False)
    renderer_point = Render(img_size=img_size, focal=focal_point, device=device, is_pointcloud=True)
    
    if cam_type == 'generated':
        pattern_name = vo_path.split('/')[-1].split('.')[0]
        RTs = generate_vo(renderer_point, pattern_name)
    else:
        camera_poses = np.loadtxt(vo_path).reshape(-1, 3, 4)
        RTs = [(pose[:, :3], pose[:, 3]) for pose in camera_poses]

    if len(RTs) < video_length:
        interpolated_poses = renderer.interpolate_slerp(RTs, T=video_length)
    else:
        interpolated_poses = RTs
    interpolated_camera_poses = []
    for i, pose in enumerate(interpolated_poses):
        R = pose[0]
        T = pose[1]
        RT = np.concatenate((R, T[..., None]), 1)
        interpolated_camera_poses.append(RT)
    interpolated_camera_poses = np.array(interpolated_camera_poses)
    camera_poses = interpolated_camera_poses

    if query_path == 'none':
        query_points = generate_query(img_size, video_length)
    else:
        query_points = np.load(query_path)[:, :, :2]

    points_3d, points_depth = estimate_depth(imagepath, query_points, img_size, focal_point, output_dir)

    projected_3d_points = []
    for idx, RT in enumerate(camera_poses):
        renderer.update_RT(RT, S)
        rendered_img = renderer.render().squeeze().numpy()[:, :, :3]

        renderer_point.update_RT(RT, S)
        projected_3d_point = renderer_point.project_vs(points_3d[idx]).cpu().numpy()
        projected_3d_point_depth = renderer.get_vertical_distances(points_3d[idx])
        projected_3d_point[:, 2] = projected_3d_point_depth
        projected_3d_points.append(projected_3d_point)
        
        cv2.imwrite(os.path.join(tmp_folder, str(idx).zfill(5) + ".png"), rendered_img)

    cam_path = "./tmp/rendered_cam.mp4"
    track_path = "./tmp/rendered_tracks.npy"

    projected_3d_points = np.array(projected_3d_points)
    np.save(track_path, projected_3d_points)

    subprocess.call(
        "ffmpeg -loglevel quiet -i {} -c:v h264 -pix_fmt yuv420p -y {}".format(
            os.path.join(tmp_folder, "%05d.png"), cam_path),
        shell=True)
    shutil.rmtree(tmp_folder)

    return projected_3d_points, points_depth, cam_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # set the gpu
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    # set start idx & end idx
    parser.add_argument('--imagepath', type=str, default='./assets/boat3.jpg', help='image path')
    parser.add_argument('--vo_path', type=str, default='./cam_patterns/PanDown.txt', help='video path')
    parser.add_argument('--query_path', type=str, default='none', help='quey point path')
    parser.add_argument('--output_dir', type=str, default='./tmp', help='output dir')
    parser.add_argument('--video_length', type=int, default=16, help='camera pose length')
    parser.add_argument('--cam_type', type=str, default='generated', help='generate or user-provided')

    args = parser.parse_args()

    # set the gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    os.makedirs(args.output_dir, exist_ok=True)

    tmp_folder = './tmp_cam'
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=False)

    img = Image.open(args.imagepath).convert('RGB')
    img_size = (img.size[1], img.size[0])
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    S = np.eye(3)
    focal_point = 500
    uvmap_obj_path = './mesh/world_envelope.obj'

    renderer = Render(img_size=img_size, device=device, uvmap_obj_path=uvmap_obj_path, is_pointcloud=False)
    renderer_point = Render(img_size=img_size, focal=focal_point, device=device, is_pointcloud=True)
    
    if args.cam_type == 'generated':
        pattern_name = args.vo_path.split('/')[-1].split('.')[0]
        RTs = generate_vo(renderer_point, pattern_name)
    else:
        camera_poses = np.loadtxt(args.vo_path).reshape(-1, 3, 4)
        RTs = [(pose[:, :3], pose[:, 3]) for pose in camera_poses]

    if len(RTs) < args.video_length:
        interpolated_poses = renderer.interpolate_slerp(RTs, T=args.video_length)
    else:
        interpolated_poses = RTs
    interpolated_camera_poses = []
    for i, pose in enumerate(interpolated_poses):
        R = pose[0]
        T = pose[1]
        RT = np.concatenate((R, T[..., None]), 1)
        interpolated_camera_poses.append(RT)
    interpolated_camera_poses = np.array(interpolated_camera_poses)
    camera_poses = interpolated_camera_poses

    if args.query_path == 'none':
        query_points = generate_query(img_size, args.video_length)
    else:
        query_points = np.load(args.query_path)[:, :, :2]

    points_3d, _ = estimate_depth(args.imagepath, query_points, img_size, focal_point, args.output_dir)

    projected_3d_points = []
    for idx, RT in enumerate(camera_poses):
        renderer.update_RT(RT, S)
        rendered_img = renderer.render().squeeze().numpy()[:, :, :3]

        renderer_point.update_RT(RT, S)
        projected_3d_point = renderer_point.project_vs(points_3d[idx]).cpu().numpy()
        projected_3d_point_depth = renderer.get_vertical_distances(points_3d[idx])
        projected_3d_point[:, 2] = projected_3d_point_depth
        projected_3d_points.append(projected_3d_point)
        
        cv2.imwrite(os.path.join(tmp_folder, str(idx).zfill(5) + ".png"), rendered_img)

    cam_path = "./tmp/rendered_cam.mp4"
    track_path = "./tmp/rendered_tracks.npy"

    projected_3d_points = np.array(projected_3d_points)
    np.save(track_path, projected_3d_points)

    subprocess.call(
        "ffmpeg -loglevel quiet -i {} -c:v h264 -pix_fmt yuv420p -y {}".format(
            os.path.join(tmp_folder, "%05d.png"), cam_path),
        shell=True)
    shutil.rmtree(tmp_folder)