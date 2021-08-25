## GBNet : GBNet: Gradient Boosting Network for Monocular Depth Estimation
## Install

```bash
git clone https://github.com/sejong-rcv/GBNet
cd GBNet
# if you want to use docker (recommended)
make docker-build
make docker-start-interactive
```


## Datasets

### Dense Depth for Autonomous Driving (DDAD)


```bash
curl -s https://tri-ml-public.s3.amazonaws.com/github/DDAD/datasets/DDAD.tar | tar -xv -C /data/datasets/
```
## Training

```bash
python3 scripts/train.py <config.yaml or checkpoint.ckpt>
```


## Evaluation


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
