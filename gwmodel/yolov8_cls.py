import os
import cv2
import numpy as np

from typing import List, Union

import torch

# Define the specific YOLOv8 detection model class
class YOLOv8_CLS:
    def __init__(self, config: dict):
        self.model_path = config['model']['path']
        self.names = {int(k):v for k,v in config['model']['names'].items()}
        self.nc = len(self.names)
        
        self.input_shape = (config['model']['height'],config['model']['width']) # shape: h,w

        _conf_thres = config['params'].get('conf_thres', None)
        if _conf_thres:
            assert 0 <= _conf_thres <= 1, f"Invalid Confidence threshold {_conf_thres}, valid values are between 0.0 and 1.0"
            self.criteria = _conf_thres
        else:
            _top = config['params'].get('top', None)
            
            if _top:
                assert isinstance(_top, int), f"Invalid top value {_top}, valid values are integer between 1 to {self.nc}"
                assert 0 <= _top <= self.nc, f"Invalid top value {_top}, valid values are integer between 1 to {self.nc}"
                self.criteria = _top
            else:
                self.criteria = 5

        self.platform = config['platform']
        self.device_id = config['device_id']

        if self.platform == 'ASCEND':
            from acllite_resource import AclLiteResource
            from acllite_model import AclLiteModel

            self.device = f'npu:{self.device_id}'

            self.resource = AclLiteResource(device_id=self.device_id)

            self.resource.init()

            self.model = AclLiteModel(self.model_path)
        else:
            from ultralytics import YOLO

            if self.device_id < torch.cuda.device_count():
               self.device = f'cuda:{self.device_id}'
            else:
               self.device = 'cpu'

            self.model=YOLO(self.model_path)
            #self.model.eval()  # Set the model to evaluation mode

    def release(self):
        if self.platform == 'ASCEND':
            del self.model
            del self.resource
        else:
            pass

    def preprocess_input(self, imgfile):
        if self.platform == 'ASCEND':
            from PIL import Image
            import torchvision.transforms as transforms
            #import torch_npu
            
            image=Image.open(imgfile)

            transform=transforms.Compose([transforms.Resize(size=self.input_shape),
                                          transforms.CenterCrop(size=self.input_shape),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.,0.,0.], std=[1.,1.,1.])
                                         ])

            return np.array(transform(image))
        else:
            pass

    def predict(self, input_data):
        """Make predictions based on the input data."""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
            
        if self.platform == 'ASCEND':
            prediction = self.model.execute([input_data])[0]
            prediction = np.squeeze(prediction)
            print(f'predictions type:{type(prediction)}, shape: {prediction.shape}')
        else:
            pass

        return prediction

    def postprocess_output(self, prediction, format='json'):
        """Process the model's predictions."""
        if self.platform == "ASCEND":
            if isinstance(self.criteria,int):
                # Get the indices of the top 5 probabilities
                _top_indices = np.argsort(prediction)[-self.criteria:][::-1]  # Sort and get the top 5 indices

                # Extract the top 5 probabilities and their corresponding class IDs
                _top_probabilities = prediction[_top_indices]
                _top_classes = _top_indices.tolist()  # Assuming class IDs are the same as the indices
                
                result=list(zip(_top_classes, _top_probabilities))
            elif isinstance(self.criteria,float):
                _thres_index=np.where(prediction >= self.criteria)[0]
                print(f'_thres_index: {_thres_index}')
                _thres_probabilities=prediction[_thres_index]
                print(f'_thres_probabilities: {_thres_probabilities}')
                #_filtered_keys=[self.names[i] for i in _filtered_index]
                #print(f'filtered_keys: {_filtered_keys}')
                result=list(zip(_thres_index,_thres_probabilities))
            else:
                pass

            print(f'result: {result}')

            if format == 'json':
                import json
                json_objs = []
                
                for r in result:
                    print(f'r: {type(r)}, {r}')
                    _idx, _poss = r
                    print(f'r1: {_idx}, {_poss}')
                    json_objs.append({
                        "class_id": int(_idx),
                        "probability": float(f'{_poss:.4f}')
                    })
                    
                return json.dumps(json_objs, indent=4)
                    
            return result
        else:
            pass

    def run_inference(self, image_file_path):
        if self.platform == 'ASCEND':
            import time

            t0 = time.time()
            preprocessed_data = self.preprocess_input(image_file_path)
            t1 = time.time()
            prediction = self.predict(preprocessed_data)
            t2 = time.time()
            postprocessed_data = self.postprocess_output(prediction)
            t3 = time.time()

            print(f'INFERENCE TIME: {t1-t0:.4f}, {t2-t1:.4f}, {t3-t2:.4f}')
            return postprocessed_data
        else:
            return self.model(image_file_path)
