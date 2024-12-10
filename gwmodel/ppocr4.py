import os
import cv2
import numpy as np
from loguru import logger

from typing import List, Union
from .onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2Img

# Define the specific YOLOv8 detection model class
class PADDLE_OCR:
    def __init__(self, config: dict):
        self.det_model = config['model'].get('det', None)
        self.rec_model = config['model'].get('rec', None)
        self.cls_model = config['model'].get('cls', None)
        
        if self.platform == 'ASCEND':
            self.device = 'cpu'
        else:
            import paddle
            
            # 检查 PaddlePaddle 是否编译了 CUDA 支持（即是否有 GPU 可用）
            if paddle.is_compiled_with_cuda():
                if self.device_id < paddle.device.cuda.device_count():
                    self.device = f'cuda:{self.device_id}'
                else:
                    self.device = 'cpu'
            else:
                self.device = 'cpu'

        self.model=ONNXPaddleOcr(det_model_dir=self.det_model, rec_model_dir=self.rec_model, cls_model_dir=self.cls_model, use_angle_cls=(not self.cls_model is None), use_gpu=('cuda' in self.device))
        logger.info(f'Paddle OCR Model Loaded')

    def release(self):
        if self.platform == 'ASCEND':
            pass
        else:
            pass
        logger.info(f'Paddle OCR Model Released')

    def run_inference(self, image_file_path):
        return self.model.ocr(image_file_path, det=self.det_model is not None, rec=self.rec_model is not None, cls=self.cls_model is not None)


if __name__ == "__main__":
    import json
    import time
    
    config_path='config/ppocr4.json'
    with open(config_path, 'r') as file:
        config = json.load(file) 

    config['platform'] = 'ASCEND'
    config['device_id'] = 0

    model=PADDLE_OCR(config)
    
    # To test preprocessing and infering
    imgfile='data/txt.jpg'
    results = model.run_inference(imgfile)
    
    sav2Img(imgfile, results)
