# Perception-as-Control
Official implementation of "Perception-as-Control: Fine-grained Controllable Image Animation with 3D-aware Motion Representation" (ICCV 2025)

**Perception-as-Control: Fine-grained Controllable Image Animation with 3D-aware Motion Representation**<br>
[Yingjie Chen](https://chen-yingjie.github.io/), 
[Yifang Men](https://menyifang.github.io/), 
[Yuan Yao](mailto:yaoy92@gmail.com), 
[Miaomiao Cui](mailto:miaomiao.cmm@alibaba-inc.com),
[Liefeng Bo](https://scholar.google.com/citations?user=FJwtMf0AAAAJ&hl=en)<br>

<p align="center">
<a href="https://arxiv.org/abs/2501.05020"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
<a href="https://chen-yingjie.github.io/projects/Perception-as-Control/index.html"><img src="https://img.shields.io/badge/Project-Website-red"></a>
<a href="https://www.youtube.com/embed/r2DszXAqbRE"><img src="https://img.shields.io/static/v1?label=Demo&message=Video&color=orange"></a>
<a href="" target='_blank'>
<img src="https://visitor-badge.laobi.icu/badge?page_id=TODO" />
</a>
</p>

<p align="center">
<img src="assets/teaser.png" width="88%" />
</p>

## 💡 Abstract
Motion-controllable image animation is a fundamental task with a wide range of potential applications. Recent works have made progress in controlling camera or object motion via the same 2D motion representations or different control signals, while they still struggle in supporting collaborative camera and object motion control with adaptive control granularity. To this end, we introduce 3D-aware motion representation and propose an image animation framework, called Perception-as-Control, to achieve fine-grained collaborative motion control. Specifically, we construct 3D-aware motion representation from a reference image, manipulate it based on interpreted user intentions, and perceive it from different viewpoints. In this way, camera and object motions are transformed into intuitive, consistent visual changes. Then, the proposed framework leverages the perception results as motion control signals, enabling it to support various motion-related video synthesis tasks in a unified and flexible way. Experiments demonstrate the superiority of the proposed method.

## 🔥 Updates
- (2025-08-04) A gradio demo is released.
- (2025-06-27) Our work has been accepted by ICCV 2025 🎉🎉🎉.
- (2025-03-31) We release the inference code and model weights of Perception-as-Control.
- (2025-03-10) We update a new version of paper with more details.
- (2025-01-09) The project page, demo video and technical report are released. The full paper version with more details is in process.

## 📑 TODO List
  - [x] Release inference code and model weights
  - [x] Provide a Gradio demo
  - [ ] Release training code

## Usage
### Environment
```shell
$ pip install -r requirements.txt
```
### Pretrained Weights
1. Download [pretrained weights](https://drive.google.com/drive/folders/1ZncmHG9K_n1BjGhVzQemomWzxQ60bYqg?usp=sharing) and put them in `$INSTALL_DIR/pretrained_weights`.

2. Download pretrained weight of based models and put them in `$INSTALL_DIR/pretrained_weights`:
  - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
  - [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder)

The pretrained weights are organized as follows:
```text
./pretrained_weights/
|-- denoising_unet.pth
|-- reference_unet.pth
|-- cam_encoder.pth
|-- obj_encoder.pth
|-- sd-vae-ft-mse
|   |-- ...
|-- sd-image-variations-diffusers
|   |-- ...
```

### Inference
```shell
$ python inference.py
```
The results will be saved in `$INSTALL_DIR/outputs`.

### Run Gradio Demo
```shell
$ python run_gradio.py
```

## 🎥 Demo 

### Fine-grained collaborative motion control
<table class="center">
    
<tr>
    <p>Camera Motion Control</p>
    <td width=33% style="border: none">
      <video controls autoplay loop src="https://github.com/user-attachments/assets/fb3aa7af-63dd-4cd1-a4a0-61dc95124b71" muted="false"></video>
    </td>
    <p>Object Motion Control</p>
    <td width=33% style="border: none">
      <video controls autoplay loop src="https://github.com/user-attachments/assets/a5e6d1a0-8116-4815-b301-9c8cd0c0039f" muted="false"></video>
    </td>
    <p>Collaborative Motion Control</p>
    <td width=33% style="border: none">
      <video controls autoplay loop src="https://github.com/user-attachments/assets/484d1324-9862-4ad8-87cb-47a5a173bcc6" muted="false"></video>
    </td>
</tr>

<tr>
    <td width=33% style="border: none">
      <video controls autoplay loop src="https://github.com/user-attachments/assets/14d73cc0-68ac-4f94-b239-b140f8cc3967" muted="false"></video>
    </td>
    <td width=33% style="border: none">
      <video controls autoplay loop src="https://github.com/user-attachments/assets/bec36750-309c-4f7c-85f4-fa18416ac09d" muted="false"></video>
    </td>
    <td width=33% style="border: none">
      <video controls autoplay loop src="https://github.com/user-attachments/assets/beb8f2be-3e6b-41a5-b605-1e7bda7902ff" muted="false"></video>
    </td>
</tr>

</table>


### Potential applications

<table class="center">
    
<tr>
    <td width=50% style="border: none">
      <p>Motion Generation</p>
      <video controls autoplay loop src="https://github.com/user-attachments/assets/cfc18a6a-d155-4b17-a9ec-5b72ca0610f6" muted="false"></video>
    </td>
    <td width=50% style="border: none">
      <p>Motion Clone</p>
      <video controls autoplay loop src="https://github.com/user-attachments/assets/7434e8e6-4e3a-46c8-81b9-d0033f37dc77" muted="false"></video>
    </td>
</tr>

<tr>
    <td width=50% style="border: none">
      <p>Motion Transfer</p>
      <video controls autoplay loop src="https://github.com/user-attachments/assets/fc8fff49-686d-43dd-bbb5-9b3ff4a04621" muted="false"></video>
    </td>
    <td width=50% style="border: none">
      <p>Motion Editing</p>
      <video controls autoplay loop src="https://github.com/user-attachments/assets/f2bf7cd0-0d68-4f90-9e4d-5c5ce0165c21" muted="false"></video>
    </td>
</tr>

</table>

For more details, please refer to our [project page](https://chen-yingjie.github.io/projects/Perception-as-Control/index.html).

## 🔗 Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{chen2025perception,
  title={Perception-as-Control: Fine-grained Controllable Image Animation with 3D-aware Motion Representation},
  author={Chen, Yingjie and Men, Yifang and Yao, Yuan and Cui, Miaomiao and Bo, Liefeng},
  journal={arXiv preprint arXiv:2501.05020},
  website={https://chen-yingjie.github.io/projects/Perception-as-Control/index.html},
  year={2025}}
```

## Acknowledgements

We would like to thank the contributors to [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone), [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2), [SpaTracker](https://github.com/henry123-boy/SpaTracker), [Tartanvo](https://github.com/castacks/tartanvo), [diffusers](https://github.com/huggingface/diffusers) for their open research and exploration.