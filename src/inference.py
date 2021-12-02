import os
import shutil
import subprocess
from pathlib import Path

output_dir = "projects/fs_vid2vid/output/face_forensics"


def inference(image_path, video_path):

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    subprocess.run(
        [
            "python",
            "inference.py",
            "--single_gpu",
            "--num_worker",
            "0",
            "--config",
            config_path,
            "--output_dir",
            output_dir,
        ]
    )