"""
Central configuration for the evaluation scripts.

Every value marked  >>> EDIT  should be checked against your paper / running system
before you report numbers. All scripts import this file, so change things here once.
"""

# ----------------------------------------------------------------------------
# 1) Column names of the frame-level table (one row per video frame)
# ----------------------------------------------------------------------------
SUBJECT_COL = "subject_id"   # volunteer / participant id
VIDEO_COL = "video_id"       # unique id per recording (= per trial)
FRAME_COL = "frame_idx"      # frame number inside the video (0, 1, 2, ...)
TIME_COL = "timestamp"       # seconds; optional -> computed as frame_idx / FPS
LABEL_COL = "label"          # per-frame class label (name or integer id)

# >>> EDIT  Landmark column naming. Default: lm23_y = y of MediaPipe landmark 23.
#           axis is one of x, y, z, v (v = visibility)
LANDMARK_COL_FMT = "lm{idx}_{axis}"

# >>> EDIT  Columns fed to the GRU, in the order the model expects.
#           None = every numeric column whose name starts with FEATURE_PREFIX,
#           kept in the order they appear in the file.
FEATURE_COLS = None
FEATURE_PREFIX = "lm"

# >>> EDIT  Class names in the exact order of the GRU's softmax outputs.
CLASS_NAMES = ["normal", "loss_of_balance", "fallen"]
# >>> EDIT  Classes whose probability counts as a "fall warning" in the decision rule.
POSITIVE_CLASSES = ["loss_of_balance"]

# ----------------------------------------------------------------------------
# 2) Video
# ----------------------------------------------------------------------------
FPS = 30.0          # >>> EDIT  recording frame rate
FRAME_W = 1280      # >>> EDIT  image width in pixels
FRAME_H = 720       # >>> EDIT  image height in pixels

# ----------------------------------------------------------------------------
# 3) GRU windows and training
# ----------------------------------------------------------------------------
WINDOW = 15             # frames per GRU input sequence
STRIDE = 1              # step between consecutive windows
WINDOW_LABEL = "last"   # "last" = label of the newest frame (matches online use) | "majority"
STANDARDIZE = False     # >>> EDIT  True if your pipeline z-scores features (fit on training fold only)

EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 8
VAL_FRACTION = 0.15     # share of TRAINING subjects (or videos) used for early stopping
CLASS_WEIGHT = True     # balance classes inside each training fold
SEED = 42
VIDEO_FOLDS = 5         # folds for the video-independent protocol

# ----------------------------------------------------------------------------
# 4) Decision rule of the running system  (GRU + persistence + hip velocity)
# ----------------------------------------------------------------------------
P_THRESH = 0.5          # GRU positive probability threshold
PERSIST_FRAMES = 3      # consecutive GRU-positive frames required
# "gru"      : N consecutive GRU-positive frames AND velocity >= V_THRESH at the current frame
# "combined" : N consecutive frames in which BOTH conditions hold
PERSIST_ON = "gru"      # >>> EDIT  to match how your system combines the two rules
V_THRESH = 1.5          # in the units of VELOCITY_MODE (1.5 m/s in the paper, Eq. 10)
VEL_WINDOW = 1          # velocity condition uses the max over the last VEL_WINDOW frames

# ----------------------------------------------------------------------------
# 5) Hip descent velocity (Eq. 3-4)
# ----------------------------------------------------------------------------
# "meters" : normalized image y  x  FRAME_H_METERS   -> m/s (needs camera calibration)
# "torso"  : pixels / torso length in pixels          -> torso-lengths per second (TL/s),
#            self-calibrating: independent of camera distance and body size
# "pixels" : pixels per second (for reference only)
VELOCITY_MODE = "meters"
VEL_LAG_FRAMES = 1      # v(t) = (y(t) - y(t-lag)) / (time(t) - time(t-lag))
FRAME_H_METERS = 2.7    # >>> EDIT  vertical real-world span (m) of the image at the subject's distance
TORSO_REF_WIN = 30      # frames in the running median of torso length ("torso" mode)
TORSO_REF_LAG = 15      # the reference ends this many frames before t, so the fall itself
                        # does not shrink the reference
MIN_VISIBILITY = 0.5    # hip landmarks below this visibility -> velocity treated as unknown

# ----------------------------------------------------------------------------
# 6) Trial-level evaluation (lead time / false alarms)
# ----------------------------------------------------------------------------
MAX_PRE_IMPACT_S = 2.0   # alerts earlier than this before impact are NOT a detection
                         # (counted as an early false alarm) when onset_frame is not annotated
ONSET_TOL_S = 0.5        # if onset_frame is annotated: valid alerts start at onset - ONSET_TOL_S
MAX_POST_IMPACT_S = 1.0  # alerts later than this after impact -> missed fall
REFRACTORY_S = 5.0       # alerts closer than this are merged into one alarm event
LEAD_TARGET_MS = (100, 300)

# ----------------------------------------------------------------------------
# 7) Threshold sensitivity grid (ablation.py)
# ----------------------------------------------------------------------------
GRID_PERSIST = [1, 2, 3, 4, 5, 6]
GRID_P = [0.5, 0.6, 0.7, 0.8, 0.9]
GRID_V = {
    "meters": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
    "torso": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
    "pixels": [150, 250, 350, 450, 550, 650, 800, 1000],
}
VEL_UNITS = {"meters": "m/s", "torso": "TL/s", "pixels": "px/s"}
