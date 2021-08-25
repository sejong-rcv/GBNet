## GBNet : GBNet: Gradient Boosting Network for Monocular Depth Estimation
## Install

You need a machine with recent Nvidia drivers and a GPU with at least 6GB of memory (more for the bigger models at higher resolution). We recommend using docker (see [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker) instructions) to have a reproducible environment. To setup your environment, type in a terminal (only tested in Ubuntu 18.04):

```bash
git clone https://github.com/TRI-ML/packnet-sfm.git
cd packnet-sfm
# if you want to use docker (recommended)
make docker-build
```

We will list below all commands as if run directly inside our container. To run any of the commands in a container, you can either start the container in interactive mode with `make docker-start-interactive` to land in a shell where you can type those commands, or you can do it in one step:

```bash
# single GPU
make docker-run COMMAND="some-command"
# multi-GPU
make docker-run-mpi COMMAND="some-command"
```

For instance, to verify that the environment is setup correctly, you can run a simple overfitting test:

```bash
# download a tiny subset of KITTI
curl -s https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/datasets/KITTI_tiny.tar | tar xv -C /data/datasets/
# in docker
make docker-run COMMAND="python3 scripts/train.py configs/overfit_kitti.yaml"
```
## Datasets

Datasets are assumed to be downloaded in `/data/datasets/<dataset-name>` (can be a symbolic link).

### Dense Depth for Autonomous Driving (DDAD)

Together with PackNet, we introduce **Dense Depth for Automated Driving** ([DDAD](https://github.com/TRI-ML/DDAD)): a new dataset that leverages diverse logs from TRI's fleet of well-calibrated self-driving cars equipped with cameras and high-accuracy long-range LiDARs.  Compared to existing benchmarks, DDAD enables much more accurate 360 degree depth evaluation at range, see the official [DDAD repository](https://github.com/TRI-ML/DDAD) for more info and instructions. You can also download DDAD directly via:

```bash
curl -s https://tri-ml-public.s3.amazonaws.com/github/DDAD/datasets/DDAD.tar | tar -xv -C /data/datasets/
```
## Training

PackNet can be trained from scratch in a fully self-supervised way (from video only, cf. [CVPR'20](#cvpr-packnet)), in a semi-supervised way (with sparse lidar using our reprojected 3D loss, cf. [CoRL'19](#corl-ssl)), and it can also use a fixed pre-trained semantic segmentation network to guide the representation learning further (cf. [ICLR'20](#iclr-semguided)).

Any training, including fine-tuning, can be done by passing either a `.yaml` config file or a `.ckpt` model checkpoint to [scripts/train.py](./scripts/train.py):

```bash
python3 scripts/train.py <config.yaml or checkpoint.ckpt>
```

If you pass a config file, training will start from scratch using the parameters in that config file. Example config files are in [configs](./configs).
If you pass instead a `.ckpt` file, training will continue from the current checkpoint state.

Note that it is also possible to define checkpoints within the config file itself. These can be done either individually for the depth and/or pose networks or by defining a checkpoint to the model itself, which includes all sub-networks (setting the model checkpoint will overwrite depth and pose checkpoints). In this case, a new training session will start and the networks will be initialized with the model state in the `.ckpt` file(s). Below we provide the locations in the config file where these checkpoints are defined:

```yaml
checkpoint:
    # Folder where .ckpt files will be saved during training
    filepath: /path/to/where/checkpoints/will/be/saved
model:
    # Checkpoint for the model (depth + pose)
    checkpoint_path: /path/to/model.ckpt
    depth_net:
        # Checkpoint for the depth network
        checkpoint_path: /path/to/depth_net.ckpt
    pose_net:
        # Checkpoint for the pose network
        checkpoint_path: /path/to/pose_net.ckpt
```

Every aspect of the training configuration can be controlled by modifying the yaml config file. This include the model configuration (self-supervised, semi-supervised, loss parameters, etc), depth and pose networks configuration (choice of architecture and different parameters), optimizers and schedulers (learning rates, weight decay, etc), datasets (name, splits, depth types, etc) and much more. For a comprehensive list please refer to [configs/default_config.py](./configs/default_config.py).

## Evaluation

Similar to the training case, to evaluate a trained model (cf. above or our [pre-trained models](#models)) you need to provide a `.ckpt` checkpoint, followed optionally by a `.yaml` config file that overrides the configuration stored in the checkpoint.

```bash
python3 scripts/eval.py --checkpoint <checkpoint.ckpt> [--config <config.yaml>]
```

You can also directly run inference on a single image or folder:

```bash
python3 scripts/infer.py --checkpoint <checkpoint.ckpt> --input <image or folder> --output <image or folder> [--image_shape <input shape (h,w)>]
```

## Models

### DDAD

| Model | Abs.Rel. | Sqr.Rel | RMSE | RMSElog | d < 1.25 |
| :--- | :---: | :---: | :---: |  :---: |  :---: |
| ResNet18, Self-Supervised, 384x640, ImageNet &rightarrow; DDAD (D) | 0.227 | 11.293 | 17.368 | 0.303 | 0.758 |
| PackNet,  Self-Supervised, 384x640, DDAD (D) | 0.173 | 7.164 | 14.363 | 0.249 | 0.835 |
| GBNet, Self-Supervised, 384x640, ImageNet &rightarrow; DDAD (D) | 0.148 | 3.329 | 14.471 | 0.244 | 0.818 |
| GBNet,  Semi-Supervised, 384x640, DDAD (D) | 0.124 | 2.476 | 13.276 | 0.220 | 0.846 |

## References

<a id="cvpr-packnet"> </a>
**3D Packing for Self-Supervised Monocular Depth Estimation (CVPR 2020 oral)** \
*Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos and Adrien Gaidon*, [**[paper]**](https://arxiv.org/abs/1905.02693), [**[video]**](https://www.youtube.com/watch?v=b62iDkLgGSI)

```
@inproceedings{packnet,
  author = {Vitor Guizilini and Rares Ambrus and Sudeep Pillai and Allan Raventos and Adrien Gaidon},
  title = {3D Packing for Self-Supervised Monocular Depth Estimation},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  primaryClass = {cs.CV}
  year = {2020},
}
```
