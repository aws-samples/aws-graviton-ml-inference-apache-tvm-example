from tvm.driver import tvmc
import os

MOUNT_DIR = '/sample/data/'
model_filename = MOUNT_DIR + 'resnet50-v1-7.onnx'
records_filename = MOUNT_DIR + 'tune_resnet50_1_3_224_224.json'


def tune_model():
    model = tvmc.load(
        model_filename,
        shape_dict={'data' : [1, 3, 224, 224]}
    )

    # Graviton 3
    tvmc.tune(
        model,
        target="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mcpu=neoverse-512tvb",
        enable_autoscheduler = True,
        tuning_records=records_filename
    )

tune_model()
