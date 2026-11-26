"""
Synthetic MediaPipe-like landmark data, ONLY to test that the scripts run end-to-end.
The numbers it produces mean nothing - replace with your real data.

Writes data_demo/frames.csv (33 landmarks x (x, y, z, visibility) per frame), data_demo/trials.csv
(fall/ADL, scenario, ground-truth onset and impact frames) and data_demo/subjects.csv.

  python make_demo_data.py --subjects 8 --out data_demo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

# standing template: (x lateral, y height) in units of stature, origin = ankle midpoint.
# MediaPipe "left" = the person's left = image right for a person facing the camera -> +x
BASE = np.array([
    (0.00, 0.935), (0.015, 0.95), (0.025, 0.952), (0.035, 0.95), (-0.015, 0.95), (-0.025, 0.952),
    (-0.035, 0.95), (0.055, 0.94), (-0.055, 0.94), (0.015, 0.915), (-0.015, 0.915),
    (0.12, 0.82), (-0.12, 0.82), (0.15, 0.64), (-0.15, 0.64), (0.16, 0.48), (-0.16, 0.48),
    (0.165, 0.44), (-0.165, 0.44), (0.16, 0.435), (-0.16, 0.435), (0.15, 0.45), (-0.15, 0.45),
    (0.065, 0.53), (-0.065, 0.53), (0.07, 0.29), (-0.07, 0.29), (0.07, 0.04), (-0.07, 0.04),
    (0.075, 0.01), (-0.075, 0.01), (0.06, 0.0), (-0.06, 0.0),
])
UPPER = np.arange(0, 23)       # face, shoulders, arms
UPPER_HIPS = np.arange(0, 25)  # + hips
KNEES = np.array([25, 26])
FALLS = ["fall_left", "fall_right", "fall_forward", "fall_backward"]
ADLS = ["walk", "sit_fast", "squat_pick", "bend", "lie_down", "jump"]


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def rotate(P, th, s, plane, idx=None, pivot_y=0.0):
    """Rotate landmarks about a horizontal axis at height pivot_y. plane='side' (image plane) or 'depth'."""
    Q = P.copy()
    idx = np.arange(len(P)) if idx is None else idx
    a = 0 if plane == "side" else 2
    u, y = P[idx, a], P[idx, 1] - pivot_y
    c, sn = np.cos(th), np.sin(th)
    Q[idx, a] = u * c + s * y * sn
    Q[idx, 1] = pivot_y - s * u * sn + y * c
    return Q


def pendulum(rng, H, speed):
    """Inverted-pendulum fall from loss of balance to impact; returns time (s) and angle (rad)."""
    th, om = rng.uniform(0.03, 0.08), rng.uniform(0.3, 0.8)
    g_l = 9.81 / (0.6 * H * rng.uniform(1.0, 1.4)) * speed
    damp, th_imp = rng.uniform(0.0, 1.0), np.deg2rad(rng.uniform(78, 84))
    dt, t, ts, ths = 1 / 600, 0.0, [0.0], [th]
    while th < th_imp and t < 3:
        om += (g_l * np.sin(th) - damp * om) * dt
        th += om * dt
        t += dt
        ts.append(t)
        ths.append(th)
    return np.array(ts), np.array(ths)


class Subject:
    def __init__(self, sid, rng):
        self.sid, self.rng = sid, rng
        self.H = rng.uniform(1.55, 1.85)
        tm = BASE.copy()
        tm[:, 0] *= rng.uniform(0.9, 1.1)
        tm[13:23, 0] += np.sign(tm[13:23, 0]) * rng.uniform(-0.02, 0.03)
        self.P0 = np.c_[tm * self.H, np.zeros(len(tm))]
        self.frame_h_m = rng.uniform(2.2, 3.4)          # camera distance differs per subject
        self.ppm = C.FRAME_H / self.frame_h_m
        self.D, self.cam_h = rng.uniform(3.0, 4.5), rng.uniform(0.9, 1.3)
        self.floor_px = C.FRAME_H * rng.uniform(0.86, 0.95)
        self.cx_px = C.FRAME_W * rng.uniform(0.4, 0.6)
        self.noise = rng.uniform(0.0015, 0.004)
        self.speed = rng.uniform(0.85, 1.25)

    def project(self, P, x_shift):
        k = self.D / (self.D - np.clip(P[:, 2], -1.5, 1.5))
        horizon = self.floor_px - self.cam_h * self.ppm
        x = (self.cx_px + (P[:, 0] + x_shift) * self.ppm * k) / C.FRAME_W
        y = (horizon + (self.cam_h - P[:, 1]) * self.ppm * k) / C.FRAME_H
        z = -(P[:, 2] - P[[23, 24], 2].mean()) * self.ppm / C.FRAME_W
        return x, y, z


def floor_clip(P, H):
    P[:, 1] = np.maximum(P[:, 1], 0.01 * H)
    return P


def gen_fall(S: Subject, kind, rng):
    fps = C.FPS
    ts, ths = pendulum(rng, S.H, S.speed)
    t_fall = ts[-1]
    n_pre, n_post = int(rng.uniform(1.2, 2.5) * fps), int(rng.uniform(1.5, 2.0) * fps)
    n = n_pre + int(np.ceil(t_fall * fps)) + n_post
    tt = np.arange(n) / fps - n_pre / fps                 # time relative to onset
    theta = np.zeros(n)
    m = (tt >= 0) & (tt <= t_fall)
    theta[m] = np.interp(tt[m], ts, ths)
    after = tt > t_fall
    theta[after] = ths[-1] + (np.deg2rad(88) - ths[-1]) * (1 - np.exp(-(tt[after] - t_fall) / 0.05))
    onset, impact = n_pre, int(np.searchsorted(tt, t_fall - 1e-9))
    labels = np.zeros(n, int)
    labels[onset:impact] = 1
    labels[impact:] = 2
    s = 1 if kind in ("fall_right", "fall_forward") else -1
    plane = "side" if kind in ("fall_left", "fall_right") else "depth"
    walking = plane == "side" and rng.random() < 0.5
    v = rng.uniform(0.6, 1.0) * s
    x0 = -v * n_pre / fps / 2 if walking else rng.uniform(-0.3, 0.3)
    poses, shifts = [], []
    for i in range(n):
        t = i / fps
        P = S.P0.copy()
        if walking and i < onset:
            P[:, 1] += 0.012 * S.H * np.sin(2 * np.pi * 1.8 * t)
            P[KNEES, 2] += 0.08 * np.sin(2 * np.pi * 0.9 * t) * np.array([1, -1])
        P = floor_clip(rotate(P, theta[i], s, plane), S.H)
        poses.append(P)
        shifts.append(x0 + (v * min(t, onset / fps) if walking else 0.01 * np.sin(2 * np.pi * 0.25 * t)))
    return poses, shifts, labels, onset, impact


def gen_adl(S: Subject, kind, rng):
    fps, H = C.FPS, S.H
    pre, post = rng.uniform(1.0, 2.0), rng.uniform(1.5, 2.0)
    dur = {"walk": rng.uniform(2.0, 3.0), "sit_fast": rng.uniform(0.45, 0.8) / S.speed,
           "squat_pick": rng.uniform(0.35, 0.6) / S.speed, "bend": rng.uniform(0.6, 1.0),
           "lie_down": rng.uniform(2.5, 3.5), "jump": 1.0}[kind]
    total = pre + dur + post + (1.2 if kind in ("squat_pick", "bend", "jump") else 0)
    n = int(total * fps)
    v = rng.choice([-1, 1]) * rng.uniform(0.8, 1.2)
    x0 = -v * total / 2 if kind == "walk" else rng.uniform(-0.3, 0.3)
    s = rng.choice([-1, 1])
    jump_h = rng.uniform(0.08, 0.15, 2)
    poses, shifts = [], []
    for i in range(n):
        t = i / fps
        u = t - pre
        P = S.P0.copy()
        shift = x0 + 0.01 * np.sin(2 * np.pi * 0.25 * t)
        if kind == "walk":
            P[:, 1] += 0.012 * H * np.sin(2 * np.pi * 1.8 * t)
            P[KNEES, 2] += 0.08 * np.sin(2 * np.pi * 0.9 * t) * np.array([1, -1])
            shift = x0 + v * t
        elif kind == "sit_fast":
            q = smoothstep(u / dur)
            P[UPPER_HIPS, 1] -= 0.26 * H * q
            P[KNEES, 2] += 0.25 * H * q
            P[UPPER, 2] += 0.08 * H * q
        elif kind == "squat_pick":
            q = smoothstep(u / dur) - smoothstep((u - dur - 0.6) / dur)
            P[UPPER_HIPS, 1] -= 0.25 * H * q
            P[KNEES, 2] += 0.22 * H * q
            P = rotate(P, np.deg2rad(35) * q, 1, "depth", UPPER, pivot_y=P[23, 1])
        elif kind == "bend":
            q = smoothstep(u / dur) - smoothstep((u - dur - 0.5) / dur)
            P[UPPER_HIPS, 1] -= 0.03 * H * q
            P = rotate(P, np.deg2rad(80) * q, 1, "depth", UPPER, pivot_y=P[23, 1])
        elif kind == "lie_down":
            P = rotate(P, np.deg2rad(85) * smoothstep(u / dur), s, "side")
        elif kind == "jump":
            for j0, h in zip((0.0, 1.2), jump_h):
                w = u - j0
                T = 2 * np.sqrt(2 * h / 9.81)
                crouch = 0.08 * H * (smoothstep(w / 0.25) - smoothstep((w - 0.25) / 0.15))
                land = 0.08 * H * (smoothstep((w - 0.4 - T) / 0.12) - smoothstep((w - 0.52 - T) / 0.3))
                P[UPPER_HIPS, 1] -= crouch + land
                if 0.4 <= w <= 0.4 + T:
                    tw = w - 0.4
                    P[:, 1] += np.sqrt(2 * 9.81 * h) * tw - 0.5 * 9.81 * tw ** 2
        poses.append(floor_clip(P, H))
        shifts.append(shift)
    return poses, shifts, np.zeros(n, int), None, None


def render(S: Subject, vid, poses, shifts, labels, rng):
    n = len(poses)
    cols = {}
    xyz = np.array([S.project(P, sh) for P, sh in zip(poses, shifts)])  # (n, 3, 33)
    jitter = np.zeros((n, 3, 33))
    e = rng.normal(0, S.noise, (n, 3, 33))
    for i in range(n):                                                  # AR(1) landmark jitter
        jitter[i] = 0.6 * jitter[i - 1] + np.sqrt(1 - 0.36) * e[i] if i else e[i]
    xyz = xyz + jitter
    vis = np.clip(np.r_[np.full(25, 0.97), np.full(8, 0.85)] + rng.normal(0, 0.03, (n, 33)), 0, 1)
    for k in range(33):
        cols[f"lm{k}_x"], cols[f"lm{k}_y"], cols[f"lm{k}_z"], cols[f"lm{k}_v"] = \
            xyz[:, 0, k], xyz[:, 1, k], xyz[:, 2, k], vis[:, k]
    df = pd.DataFrame(cols)
    df.insert(0, "label", [C.CLASS_NAMES[c] for c in labels])
    df.insert(0, "timestamp", np.arange(n) / C.FPS)
    df.insert(0, "frame_idx", np.arange(n))
    df.insert(0, "video_id", vid)
    df.insert(0, "subject_id", S.sid)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=8)
    ap.add_argument("--out", default="data_demo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames, trials, subjects = [], [], []
    for si in range(args.subjects):
        S = Subject(f"S{si + 1:02d}", rng)
        subjects.append({"subject_id": S.sid, "height_m": round(S.H, 3), "frame_h_m": round(S.frame_h_m, 3)})
        plan = [("fall", k) for k in FALLS for _ in range(2)] + [("adl", k) for k in ADLS] + \
               [("adl", "sit_fast"), ("adl", "jump")]
        for j, (ttype, kind) in enumerate(plan):
            vid = f"{S.sid}_{kind}_{j:02d}"
            if ttype == "fall":
                poses, shifts, labels, onset, impact = gen_fall(S, kind, rng)
            else:
                poses, shifts, labels, onset, impact = gen_adl(S, kind, rng)
            frames.append(render(S, vid, poses, shifts, labels, rng))
            trials.append({"video_id": vid, "subject_id": S.sid, "trial_type": ttype, "scenario": kind,
                           "onset_frame": onset, "impact_frame": impact})
    fr = pd.concat(frames, ignore_index=True)
    fr.to_csv(out / "frames.csv", index=False, float_format="%.5f")
    pd.DataFrame(trials).to_csv(out / "trials.csv", index=False)
    pd.DataFrame(subjects).to_csv(out / "subjects.csv", index=False)
    print(f"{len(fr)} frames, {len(trials)} videos, {args.subjects} subjects -> {out}/")
    print(fr["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
