import os
import torch
import numpy as np
from copy import deepcopy

from pipe.cfgs import load_cfg
from pipe.c2f_recons import Pipeline
from ops.gs.basic import Frame
from ops.trajs.basic import Traj_Base
from ops.utils import save_pic


def yaw_matrix(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    rot = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ])
    return rot


def render_view(scene, extrinsic: np.ndarray, save_fn: str):
    H = scene.frames[0].H
    W = scene.frames[0].W
    intrinsic = deepcopy(scene.frames[0].intrinsic)
    frame = Frame(H=H, W=W, intrinsic=intrinsic, extrinsic=extrinsic)
    rgb, _, _ = scene._render_RGBD(frame)
    rgb_np = rgb.detach().cpu().numpy()
    save_pic(rgb_np, save_fn, normalize=False)


def main():
    cfg = load_cfg('pipe/cfgs/basic.yaml')
    cfg.scene.input.rgb = 'data/sd_readingroom/color.png'

    scene_pth = 'data/sd_readingroom/scene.pth'
    if not os.path.exists(scene_pth):
        pipeline = Pipeline(cfg)
        pipeline()

    scene = torch.load(scene_pth)

    # estimate scene radius for translation
    base_traj = Traj_Base(scene, nframe=1)
    radius = base_traj.radius

    base_extr = np.eye(4)
    # move the camera slightly backward along the z-axis
    base_extr[:3, 3] += np.array([0, 0, -radius * 0.3])

    start_deg = -45
    end_deg = 45

    # move slightly right and rotate left (start_deg)
    extr_left = deepcopy(base_extr)
    extr_left[:3, 3] += np.array([radius * 0.1, 0, 0])
    extr_left[:3, :3] = yaw_matrix(start_deg) @ extr_left[:3, :3]
    render_view(scene, extr_left, 'data/sd_readingroom/right.png')

    # move slightly left and rotate right (end_deg)
    extr_right = deepcopy(base_extr)
    extr_right[:3, 3] += np.array([-radius * 0.1, 0, 0])
    extr_right[:3, :3] = yaw_matrix(end_deg) @ extr_right[:3, :3]
    render_view(scene, extr_right, 'data/sd_readingroom/left.png')


if __name__ == '__main__':
    main()