"""Fetch / document external assets used by the project.

Already fetched into ``data/`` (see README):
  * FireNet pretrained model + repo  -> data/firenet/
  * N-Cars test split (real events)  -> data/ncars/

This script records the exact sources and can re-fetch the small ones. The
denoising benchmarks (DVSNOISE20, DND21) are large/gated and are documented as
URLs only — download manually if you want real-data denoising numbers.
"""

from __future__ import annotations

import os
import subprocess
import sys

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

ASSETS = {
    "firenet_model": {
        "status": "fetched",
        "path": "data/firenet/firenet_1000.pth.tar",
        "source": "Google Drive id 1nBCeIF_Us-rGhCjdU5q1Ch-yrFckjZPa "
                  "(from rpg_e2vid README; the rpg.ifi.uzh.ch path 404s)",
        "note": "Scheerlinck et al., WACV 2020. 37,777 params.",
    },
    "firenet_repo": {
        "status": "fetched",
        "path": "data/firenet/rpg_e2vid (branch cedric/firenet)",
        "source": "https://github.com/cedric-scheerlinck/rpg_e2vid",
        "note": "Reference model definition (we reimplement it in firenet.py).",
    },
    "ncars": {
        "status": "fetched (test split)",
        "path": "data/ncars/n-cars_test/test/{cars,background}/*.dat",
        "source": "https://downloads.prophesee.ai/website/resources/Prophesee_Dataset_n_cars.zip",
        "note": "Sironi et al. (HATS), CVPR 2018. 8607 test files. ATIS .dat.",
    },
    "dvsnoise20": {
        "status": "URL-ONLY (large/gated — download manually)",
        "path": "data/dvsnoise20/",
        "source": "UDayton ISSL Google Drive — "
                  "aedat 1_aedat.zip id=1bIX23bppCyGvtmo4aqTLUmWwbC9wSCX3 ; "
                  "mat 2_mat.zip id=1k3ir-mXJQhQgVWvdI_ouekRZ6-yQeCYb (~7GB) ; "
                  "epm 5_epm.zip id=1omYF3ecjrQVhfpgEe6nm9FgfLo4Yax3_ . "
                  "Code: https://github.com/bald6354/edncnn",
        "note": "Baldwin et al. (EPM/EDnCNN), CVPR 2020. Real DAVIS346 BA noise + EPM labels.",
    },
    "edncnn_pretrained": {
        "status": "URL-ONLY (large — download manually; ~297 MB)",
        "path": "data/edncnn/allData_v8_preTrained.mat",
        "source": "Authors' pretrained EDnCNN, from https://github.com/bald6354/edncnn "
                  "(pretrained-model Google Drive link in the repo README).",
        "note": "Baldwin et al., CVPR 2020. Loaded by stcd/downstream/edncnn_real.py "
                "via h5py (weights-only; no pickle).",
    },
    "mlpf_weights": {
        "status": "fetched (small; vendored from the MLPF repo)",
        "path": "data/mlpf/vendor/0316_soft_4bit_alpha1_sigmoid.h5",
        "source": "https://github.com/SensorsINI/dnd_hls "
                  "(hls4ml_model_generation/0316_soft_4bit_alpha1_sigmoid.h5)",
        "note": "Rios-Navarro et al. (MLPF), CVPRW 2023. Published 98->10->1 4-bit weights; "
                "loaded by stcd/downstream/mlpf.py.",
    },
}


def main() -> None:
    print("External asset status\n" + "=" * 60)
    for key, a in ASSETS.items():
        abspath = os.path.join(DATA, a["path"].replace("data/", "", 1).split(" ")[0])
        present = os.path.exists(abspath)
        mark = "✓" if present else ("•" if "URL" in a["status"] else "✗")
        print(f"\n[{mark}] {key}  ({a['status']})")
        print(f"    path:   {a['path']}")
        print(f"    source: {a['source']}")
        print(f"    note:   {a['note']}")
    print("\nDVSNOISE20 / DND21 are not auto-downloaded (large/gated). "
          "Fetch manually with the IDs above to enable real-data denoising eval.")


if __name__ == "__main__":
    main()
