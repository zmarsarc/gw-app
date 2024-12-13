import json
from pathlib import Path

import cv2
import numpy as np
import torch
from enum import Enum

from hat.model_handler import HatDetectHatHandler
from intrusion.model_handler import IntrusionDetectIntrusionHandler
from wandering.model_handler import BehaviorDetectWanderingHandler
from lightning_rod_current_meter.model_handler import PointerMeterDetectLightningRodCurrentMeterHandler
from cabinet_meter.model_handler import IndicatorMeterDetectCabinetMeterHandler

model_classes = { "hat": HatDetectHatHandler, 
                  "intrusion": IntrusionDetectIntrusionHandler,
                  "wandering": BehaviorDetectWanderingHandler, 
                  "lightning_rod_current_meter": PointerMeterDetectLightningRodCurrentMeterHandler,
                  "cabinet_meter": IndicatorMeterDetectCabinetMeterHandler
}   

PLATFORM = ['ONNX', 'ASCEND']
    
class GWProc:
    def __init__(self, model_name, platform='ASCEND', device_id=None):
        # Validate the model type
        if model_name not in model_classes:
            raise ValueError(
                f"Invalid model type '{model_name}'. Available types: {list(self.model_classes.keys())}")

        if platform not in PLATFORM:
            raise ValueError(
                f"Invalid platform type '{platform}'. Available platforms: {PLATFORM}")

        self.model_instance = model_classes[model_name](platform=platform, device_id=device_id)

    def run_inference(self, input_image):
        """Run inference using the specific model instance."""
        return self.model_instance.run_inference(input_image)

    def release(self) -> None:
        self.model_instance.release()


# Example usage
if __name__ == "__main__":
    # Load the configuration for the specific task
    import time

    t0 = time.time()
    #model = GWProc(model_name='hat', platform='ONNX')
    #model = GWProc(model_name='intrusion', platform='ONNX')
    #model = GWProc(model_name='wandering', platform='ONNX')
    #model = GWProc(model_name='lightning_rod_current_meter', platform='ONNX')
    #model = GWProc(model_name='cabinet_meter', platform='ONNX')

    #model = GWProc(model_name='hat', platform='ASCEND', device_id=0)
    #model = GWProc(model_name='intrusion', platform='ASCEND', device_id=0)
    #model = GWProc(model_name='wandering', platform='ASCEND', device_id=0)
    #model = GWProc(model_name='lightning_rod_current_meter', platform='ASCEND', device_id=0)
    model = GWProc(model_name='cabinet_meter', platform='ASCEND', device_id=0)
    t1 = time.time()
    
    # Load an image for testing (replace with actual image path)
    #input_images = ['hat/test_case/0.jpg','hat/test_case/1.jpg','hat/test_case/2.jpg']
    #input_images = ['intrusion/test_case/0.jpg','intrusion/test_case/2.jpg','intrusion/test_case/4.jpg']
    #input_images = ['wandering/test_case/2024-10-25T17.30.16-[539,119,611,278]-8d4376beeb5048feb0cfb7ed798b6d71.png',
    #                'wandering/test_case/2024-10-25T17.30.16-[539,119,611,278]-27fb1e1bd3844b07a18f7d2aac8bbc71.png',
    #                'wandering/test_case/2024-10-25T17.30.16-[539,119,611,278]-93bfd984746c49e0b02cf33b7575bcb3.png',
    #                'wandering/test_case/2024-10-25T17.30.16-[539,119,611,278]-ef7e24fc0da949f4af6a471209904250.png',
    #                'wandering/test_case/2024-10-25T17.30.16-[539,119,611,278]-f325e0299dbd4c77ae71f5b4ebf7ad38.png',
    #                'wandering/test_case/2024-10-25T17.30.16-[539,119,611,278]-faf3ca75c7014e55ac307d8eddcf8616.png'
    #                ]
    #input_images = ['lightning_rod_current_meter/test_case/lightning_rod_current_meter2.png']
    input_images = ['cabinet_meter/test_case/cabinet_meter_20A.jpg']

    # Run inference
    results = model.run_inference(input_images)
    t2 = time.time()

    print(f'TIME: {t1-t0:.4f}, {t2-t1:.4f}')

    # Process and visualize the results (example)
    print("Inference Results:", results)

    model.release()

    print("Done!")
