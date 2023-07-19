"""DeeplabV3+ with ResNet-50-d8."""

_base_ = [
    "./deeplabv3plus-r50-d8-config-sem_seg.py",
    # "../_base_/datasets/bdd100k_512x1024.py",
    # "../_base_/default_runtime.py",
    # "../_base_/schedules/schedule_80k.py",
]
# load_from = "https://dl.cv.ethz.ch/bdd100k/drivable/models/deeplabv3+_r50-d8_512x1024_80k_drivable_bdd100k.pth"