"""Event Probability Mask (EPM) — port of the EDnCNN labelling pipeline.

Computes the per-pixel temporal log-intensity derivative Jt = ∇L·v from the APS
frames and IMU camera rotation (Baldwin et al., CVPR 2020), then the per-event
plausibility Prob = clamp(±Jt/gamma, 0, 1). This is the field-standard ground
truth EDnCNN was trained on; computing it ourselves (event-aligned) sidesteps the
mis-aligned published epm_array and lets us report literature-comparable RPMD.

Ported from the authors' MATLAB: assignJt2Events.m, ground_truth_motion.m,
loadDistortion.m, interpolateIMU.m. Camera params in data/edncnn/camera/.
"""

from __future__ import annotations

import os
import numpy as np

CAM = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "edncnn", "camera")


def load_camera():
    import h5py
    def rd(f, k):
        with h5py.File(os.path.join(CAM, f), "r") as h: return np.array(h[k]).T  # ->(H,W)
    U = rd("UandV_346.mat", "U"); V = rd("UandV_346.mat", "V")
    jac = {k: rd("jacobian_346.mat", k) for k in ("dx_du", "dx_dv", "dy_du", "dy_dv")}
    with h5py.File(os.path.join(CAM, "cameraParameters_346.mat"), "r") as h:
        K = np.array(h["#refs#"]["c"]["IntrinsicMatrix"])      # MATLAB transposed
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2] if K[0, 2] != 0 else K[2, 0]), float(K[1, 2] if K[1, 2] != 0 else K[2, 1])
    normU = (U - cx) / fx
    normV = (V - cy) / fy
    return normU, normV, jac


def _interp_gyro(imu_t_us, gyro, frame_t_us, bias_s=0.5):
    """deg/s gyro -> bias-removed, median-smoothed, interpolated to frame times."""
    from scipy.ndimage import median_filter
    order = np.argsort(imu_t_us); t = imu_t_us[order].astype(np.float64); g = gyro[order].astype(np.float64)
    bias = g[(t - t.min()) / 1e6 <= bias_s].mean()
    g = median_filter(g - bias, size=21)
    return np.interp(frame_t_us.astype(np.float64), t, g)


def compute_jt(aps_fhw, gyroX_f, gyroY_f, normU, normV, jac, integ_time_s, offset=50.0):
    """Jt volume (F,H,W). gyroX_f/gyroY_f: per-frame gyro (deg/s)."""
    F, H, W = aps_fhw.shape
    Jt = np.empty((F, H, W), np.float32)
    for f in range(F):
        ox = -np.deg2rad(gyroX_f[f]); oy = -np.deg2rad(gyroY_f[f])      # omega (z=0), negated
        U, Vn = normU, normV
        Vu = -oy + ox * U * Vn - oy * U**2                              # ground_truth_motion
        Vv = ox - oy * U * Vn + ox * Vn**2
        Vx = jac["dx_du"] * Vu + jac["dx_dv"] * Vv                      # distortion jacobian
        Vy = jac["dy_du"] * Vu + jac["dy_dv"] * Vv
        im = aps_fhw[f].astype(np.float64)
        # directional (upwind) intermediate gradient
        fwd_x = np.zeros_like(im); fwd_x[:, :-1] = im[:, 1:] - im[:, :-1]
        bwd_x = np.zeros_like(im); bwd_x[:, 1:] = im[:, 1:] - im[:, :-1]
        fwd_y = np.zeros_like(im); fwd_y[:-1, :] = im[1:, :] - im[:-1, :]
        bwd_y = np.zeros_like(im); bwd_y[1:, :] = im[1:, :] - im[:-1, :]
        Gx = np.where(Vx > 0, bwd_x, fwd_x) * np.maximum(1.0, np.abs(Vx * integ_time_s))
        Gy = np.where(Vy > 0, bwd_y, fwd_y) * np.maximum(1.0, np.abs(Vy * integ_time_s))
        denom = np.maximum(im + offset, 1.0)
        Jt[f] = (Gx / denom) * Vx + (Gy / denom) * Vy
    return Jt


def event_prob(jt_volume, x, y, closest_frame, polarity, gammaP=20.0, gammaN=20.0):
    """Per-event EPM probability = clamp(Jt/gammaP for ON / -Jt/gammaN for OFF, 0, 1)."""
    jt = jt_volume[closest_frame, y, x]
    g = np.where(polarity > 0, jt / gammaP, -jt / gammaN)
    return np.clip(g, 0.0, 1.0)
