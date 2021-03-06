## GBNet : Gradient Boosting Network for Monocular Depth Estimation

This is the reference PyTorch implementation for training and testing depth estimation models, and it based on the [**PackNet**](#cvpr-packnet) code. 
## Install

We recommend using docker.  To setup your environment, type in a terminal:

```bash
git clone https://github.com/sejong-rcv/GBNet
cd GBNet
# if you want to use docker (recommended)
make docker-build

```

if you want to run directly inside our container, you can do it in one step:

```
# single GPU
make docker-run COMMAND="some-command"
# multi-GPU
make docker-run-mpi COMMAND="some-command"
```

If you want to run any of the commands in container, you can do in one step:
```
make docker-start-interactive
```

If you want to use features related Weights & Biases (WANDB) (for experiment management/visualization), then you should create associated accounts and configure your shell with the following environment variables:

```
export WANDB_ENTITY="something"
export WANDB_API_KEY="something"
```
To enable WANDB logging, you can then set the corresponding configuration parameters in configs/<your config>.yaml (cf. configs/default_config.py for defaults and docs):
```
wandb:
    dry_run: True                                 # Wandb dry-run (not logging)
    name: ''                                      # Wandb run name
    project: os.environ.get("WANDB_PROJECT", "")  # Wandb project
    entity: os.environ.get("WANDB_ENTITY", "")    # Wandb entity
    tags: []                                      # Wandb tags
    dir: ''                                       # Wandb save folder
```
## Datasets
Datasets are assumed to be downloaded in /data/datasets/<dataset-name> (can be a symbolic link).
### Dense Depth for Autonomous Driving (DDAD)


```bash
curl -s https://tri-ml-public.s3.amazonaws.com/github/DDAD/datasets/DDAD.tar | tar -xv -C /data/datasets/
```
## Training

- To initialize the packnet, download pre-trained weight in [github](https://github.com/TRI-ML/packnet-sfm)

```bash
python3 scripts/train.py configs/train_ddad.yaml
```


## Evaluation


```bash
python3 scripts/eval.py --checkpoint <checkpoint.ckpt> --config configs/eval_ddad.yaml
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
| [GBNet, Self-Supervised, 384x640, DDAD (D)](http://multispectral.sejong.ac.kr:8080/share.cgi?ssid=0gv2Kx0) | 0.148 | 3.329 | 14.471 | 0.244 | 0.818 |
| [GBNet,  Semi-Supervised, 384x640, DDAD (D)](http://multispectral.sejong.ac.kr:8080/share.cgi?ssid=0rycygL) | 0.124 | 2.476 | 13.276 | 0.220 | 0.846 |

## References

[**GBNet**](#GBNet) is based on [**PackNet**](#cvpr-packnet) code. 

<a id="GBNet"> </a>
**GBNet : Gradient Boosting Network for Monocular Depth Estimation (ICCAS)** \
*Daechan Han and Yukyung Choi*
```
@inproceedings{GBNet,
  author = {Daechan Han and Yukyung Choi},
  title = {GBNet : Gradient Boosting Network for Monocular Depth Estimation},
  booktitle = {International Conference of Contral, Automation and Systems(ICCAS)},
  year = {2021},
}
```

<a id="cvpr-packnet"> </a>
**3D Packing for Self-Supervised Monocular Depth Estimation** \
*Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos and Adrien Gaidon*
```
@inproceedings{packnet,
  author = {Vitor Guizilini and Rares Ambrus and Sudeep Pillai and Allan Raventos and Adrien Gaidon},
  title = {3D Packing for Self-Supervised Monocular Depth Estimation},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  primaryClass = {cs.CV}
  year = {2020},
}
```
