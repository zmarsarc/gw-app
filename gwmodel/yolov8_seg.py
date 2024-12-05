import os
import cv2
import numpy as np

from typing import List, Union

import torch

# Define the specific YOLOv8 detection model class
class YOLOv8_SEG:
    def __init__(self, config: dict):
        self.model_path = config['model']['path']
        self.names = {int(k):v for k,v in config['model']['names'].items()}
        self.nc = len(self.names)
        
        self.input_shape = (config['model']['height'],config['model']['width']) # shape: h,w

        self.conf_thres = config['params'].get('conf_thres', 0.5)
        self.iou_thres = config['params'].get('iou_thres', 0.4)

        self.platform = config['platform']
        self.device_id = config['device_id']

        if self.platform == 'ASCEND':
            from acllite_resource import AclLiteResource
            from acllite_model import AclLiteModel

            self.device = f'npu:{self.device_id}'

            self.resource = AclLiteResource(device_id=self.device_id)

            self.resource.init()

            print(f'model: {self.model_path}')
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

    def preprocess_input(self, im):
        if self.platform == 'ASCEND':
            im = np.stack(self.pre_transform([im]))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous

            im = (im / 255).astype(np.float32)  # 0 - 255 to 0.0 - 1.0
            
            return im
        else:
            pass
        
    def pre_transform(self, im):
        """
        Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (list): A list of transformed images.
        """
        from utils import LetterBox

        #same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(self.input_shape) #, auto=same_shapes and self.model.pt, stride=self.model.stride)
        return [letterbox(image=x) for x in im]

    def predict(self, input_data):
        """Make predictions based on the input data."""
        if self.model is None:
            raise ValueError("Model not loaded.")
            
        if self.platform == 'ASCEND':
            predictions = self.model.execute([input_data])
            #predictions = np.squeeze(predictions)
            print(f'0. predictions len: {len(predictions)}, type:{type(predictions)}')
            for _i,_pred in enumerate(predictions):
                print(f'0.{_i} pred: {type(_pred)}, {_pred.shape}')
        else:
            pass

        return predictions

    def postprocess_output(self, predictions, img1, img0, img0_path, format='json'):
        """Process the model's predictions."""
        if self.platform == "ASCEND":
            import ops
            from results import Results
            
            """Post-processes predictions and returns a list of Results objects."""
            preds=torch.tensor(predictions[0])
            print(f'1. preds type: {type(preds)}, shape: {preds.shape}')
            pred = ops.non_max_suppression(
                preds,
                self.conf_thres,
                self.iou_thres,
                #agnostic=self.args.agnostic_nms,
                #max_det=self.args.max_det,
                #classes=self.args.classes,
                nc=self.nc,
            )[0]
            #print(f'2. pred type: {type(pred)}, shape: {pred.shape}')
            
            results = []
            proto = torch.tensor(predictions[1][0]) #[-1] if isinstance(preds[1], tuple) else preds[1]  # tuple if PyTorch model or array if exported
            #print(f'3. proto type: {type(proto)}, shape: {proto.shape}')
            #print(f'img0: {img0.shape}')
            #print(f'img1: {img1.shape}')

            if not len(pred):  # save empty boxes
                masks = None
            else:
                masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img1.shape[2:], upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(img1.shape[2:], pred[:, :4], img0.shape)
                
            #print(f'4. pred: {pred.shape}, masks: {masks.shape}')
            #print(pred[:,:6])
            #print(masks)

            #results = Results(img0, path=img0_path, names=self.names, boxes=pred[:, :6], masks=masks)
            
            
            if format == 'json':
                import json
                import base64
                
                # Convert to a more structured format for JSON
                seg_detections = []
                for _i in range(len(pred)):
                    x_min, y_min, x_max, y_max, confidence, class_id = pred[_i][:6]

                    # Encode the mask as a binary image
                    _, buffer = cv2.imencode('.png', masks[_i].numpy())  # Convert the mask to PNG format
                    binary_mask = buffer.tobytes()  # Get the binary data

                    # Encode the binary data as Base64
                    encoded_mask = base64.b64encode(binary_mask).decode('utf-8')
                    mask_data = {
                        "shape": masks[_i].shape,
                        "data": encoded_mask
                    }                    
                    
                    seg_detections.append({
                        "class_id": int(class_id),
                        "confidence": float(f'{confidence:.4f}'),
                        "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                        "mask": mask_data,  # List of keypoints
                    })
                    
                # Convert to JSON
                return json.dumps(seg_detections, indent=4)

            #results = Results(img0, path=img0_path, names=self.names, boxes=pred[:, :6], masks=masks)
            #return results

            return (pred[:,:6], masks)
        else:
            pass

    def run_inference(self, image_file_path):
        if self.platform == 'ASCEND':
            import time
            import cv2

            t0 = time.time()
            img0=cv2.imread(image_file_path)
            img1 = self.preprocess_input(img0)
            #print(f'preprocessed_data type: {type(img1)}, shape: {img1.shape}')
            t1 = time.time()
            predictions = self.predict(img1)
            t2 = time.time()
            results = self.postprocess_output(predictions, img1, img0, image_file_path)
            t3 = time.time()

            print(f'INFERENCE TIME: {t1-t0:.4f}, {t2-t1:.4f}, {t3-t2:.4f}')
            return results
        else:
            return self.model(image_file_path)
                       
if __name__ == "__main__":
    import json
    import time
    
    config_path='config/yolov8n-seg.json'
    with open(config_path, 'r') as file:
        config = json.load(file) 

    config['platform'] = 'ASCEND'
    config['device_id'] = 0

    model=YOLOv8_SEG(config)
    
    # To test preprocessing and infering
    imgfile='data/bus.jpg'
    results = model.run_inference(imgfile)
    #print(results)
    
    # To test postrocessing and infering
    """
    model.platform="ASCEND"

    t0 = time.time()
    image_file_path='data/bus.jpg'
    img0=cv2.imread(image_file_path)
    img1 = model.preprocess_input(img0)
    print(f'preprocessed_data type: {type(img1)}, shape: {img1.shape}')
    t1 = time.time()
    predicted_data = torch.load('det-preds.pt')
    results = model.postprocess_output(predicted_data, img1, img0, image_file_path)
    t2 = time.time()
    
    print(f'TIME: {t1-t0:.4f}, {t2-t1:.4f}')

    
    print('Done')
    """
