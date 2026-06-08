# Data & pretrained weights

None of the datasets or pretrained weights are included in this repository
(~34 GB). This file documents how to obtain each asset and where to place it. Run
`python scripts/download_data.py` to print the exact sources and check which
assets are present locally.

All assets below are **third-party** and retain their original licenses; we do
not redistribute them. The STCD code in this repo is MIT-licensed (see LICENSE).

## Layout expected by the scripts

```
data/
  dvsnoise20/2_mat/*.mat                       # 16 real DAVIS346 recordings
  edncnn/allData_v8_preTrained.mat             # pretrained EDnCNN (~297 MB)
  mlpf/vendor/0316_soft_4bit_alpha1_sigmoid.h5 # published MLPF weights
  firenet/firenet_1000.pth.tar                 # pretrained FireNet
  ncars/n-cars_test/test/{cars,background}/*.dat
```

## Sources

| Asset | Used by | Source | License |
|---|---|---|---|
| **DVSNOISE20** (real DAVIS346, 16 recordings) | the AUC + downstream eval | UDayton ISSL — `2_mat.zip` (~7 GB) via the Drive IDs in `scripts/download_data.py`; code at https://github.com/bald6354/edncnn | Baldwin et al., CVPR 2020 |
| **EDnCNN pretrained weights** (`allData_v8_preTrained.mat`) | `stcd/downstream/edncnn_real.py` | https://github.com/bald6354/edncnn (pretrained-model Drive link in README) | Baldwin et al., CVPR 2020 |
| **MLPF weights** (`0316_soft_4bit_alpha1_sigmoid.h5`) | `stcd/downstream/mlpf.py` | https://github.com/SensorsINI/dnd_hls (`hls4ml_model_generation/`) | Rios-Navarro et al., CVPRW 2023 |
| **FireNet** (`firenet_1000.pth.tar`) | `stcd/downstream/firenet.py` | https://github.com/cedric-scheerlinck/rpg_e2vid | Scheerlinck et al., WACV 2020 |
| **N-Cars** (test split) | `scripts/run_ncars_recognition.py` | https://www.prophesee.ai/2018/03/13/dataset-n-cars/ | Sironi et al., CVPR 2018 |

## Notes

- Weights are loaded **safely**: EDnCNN/MLPF via `h5py` (no pickle), FireNet with
  `torch.load(..., weights_only=True)`. We never run `weights_only=False` on
  external checkpoints.
- The 16-recording denoising eval (`figures/data/edncnn_real.json`) is the one
  cached result committed to the repo, so `run_pareto.py` and
  `run_edncnn_efficiency.py` regenerate their figures **without** re-downloading
  the 34 GB of raw recordings.
