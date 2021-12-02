import os
import shutil
import subprocess
from pathlib import Path

output_dir = "output"


def inference(image_path, mask_path):
    curr_dir = Path.cwd()
    model_path = os.path.join(curr_dir, "big-lama")
    in_dir = os.path.join(curr_dir, "input")
    out_dir = os.path.join(curr_dir, output_dir)

    if os.path.isdir(in_dir):
        shutil.rmtree(in_dir)
    os.makedirs(in_dir, exist_ok=True)

    image_stem = Path(image_path).stem
    image_name = image_stem + ".png"
    mask_name = image_stem + "_mask.png"

    shutil.move(image_path, os.path.join(in_dir, image_name))
    shutil.move(mask_path, os.path.join(in_dir, mask_name))

    subprocess.run(
        [
            "python",
            "bin/predict.py",
            f"model.path={model_path}",
            f"indir={in_dir}",
            f"outdir={out_dir}",
        ]
    )

    return mask_name
