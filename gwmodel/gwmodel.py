import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results

from .ppocr4 import PADDLE_OCR
from .yolov8_cls import YOLOv8_CLS
from .yolov8_det import YOLOv8_DET
from .yolov8_pose import YOLOv8_POSE
from .yolov8_seg import YOLOv8_SEG

model_classes = {"classification": YOLOv8_CLS, "detection": YOLOv8_DET,
                 "pose": YOLOv8_POSE, "segment": YOLOv8_SEG, "ppocr": PADDLE_OCR}


class GWModel:
    def __init__(self, config, device_id, platform='ASCEND'):
        self.config = config

        # Validate the model type
        model_type = self.config['model']['type']
        if model_type not in model_classes:
            raise ValueError(
                f"Invalid model type '{model_type}'. Available types: {list(self.model_classes.keys())}")

        # Add additional config params to config for creating the specific model
        self.config['platform'] = platform
        self.config['device_id'] = device_id

        self.model_instance = model_classes[model_type](self.config)

    def run_inference(self, input_image) -> Results:
        """Run inference using the specific model instance."""
        return self.model_instance.run_inference(input_image)

    def release(self) -> None:
        self.model_instance.release()


# Example usage
if __name__ == "__main__":
    # Load the configuration for the specific task
    import time

    t0 = time.time()
    model = GWModel(config_path='config/ppocr4.json',
                    device_id=0, platform='ASCEND')
    t1 = time.time()
    # Load an image for testing (replace with actual image path)
    # input_image = Path('data/txt.jpg')
    input_image = 'data/txt.jpg'

    # Run inference
    results = model.run_inference(input_image)
    t2 = time.time()

    print(f'TIME: {t1-t0:.4f}, {t2-t1:.4f}')

    # Process and visualize the results (example)
    print("Inference Results:", results)

    model.release()

    print("Done!")
