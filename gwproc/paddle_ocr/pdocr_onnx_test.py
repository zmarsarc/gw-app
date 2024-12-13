# -*- coding: utf-8 -*-
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

try:
    from .pdocr_onnx import OcrOnnx
except:
    from pdocr_onnx import OcrOnnx

s_cls = ort.InferenceSession('../assets/models/pointer_meter_detect_cabinet_meter/1/pdocr_cls.onnx',
                             providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
s_rec = ort.InferenceSession('../assets/models/pointer_meter_detect_cabinet_meter/1/pdocr_rec.onnx',
                             providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
s_det = ort.InferenceSession('../assets/models/pointer_meter_detect_cabinet_meter/1/pdocr_det.onnx',
                             providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

text_sys = OcrOnnx(cls_sess=s_cls, det_sess=s_det, rec_sess=s_rec)

img = cv2.imread(r'E:\external_cv_sanxiayunbian\services\000-multi_onnx_test\paddle_ocr\test_case\biaoji.jpg')
# img = np.asarray(Image.open('test_case/biaoji.jpg'))

res = text_sys(img)

print(res)

data = {'msg': '', 'status': '000',
        'results': [
            [{'confidence': 0.781145453453064, 'text': '07-14-2023星期开09-',
              'text_region': [[2.0, 60.0], [662.0, 60.0], [662.0, 107.0], [2.0, 107.0]]},
             {'confidence': 0.9174224138259888, 'text': 'JCQ-3E型避雷器监测器',
              'text_region': [[520.0, 504.0], [976.0, 502.0], [976.0, 544.0], [520.0, 546.0]]},
             {'confidence': 0.9650097489356995, 'text': '1.0',
              'text_region': [[654.0, 584.0], [702.0, 584.0], [702.0, 613.0], [654.0, 613.0]]},
             {'confidence': 0.8598756194114685, 'text': '1.5',
              'text_region': [[728.0, 584.0], [786.0, 584.0], [786.0, 606.0], [728.0, 606.0]]},
             {'confidence': 0.9061177372932434, 'text': '2.0',
              'text_region': [[806.0, 584.0], [862.0, 584.0], [862.0, 613.0], [806.0, 613.0]]},
             {'confidence': 0.8884006142616272, 'text': '2.5',
              'text_region': [[897.0, 593.0], [949.0, 598.0], [945.0, 632.0], [893.0, 626.0]]},
             {'confidence': 0.9951229691505432, 'text': '0.5',
              'text_region': [[564.0, 604.0], [610.0, 591.0], [618.0, 621.0], [572.0, 633.0]]},
             {'confidence': 0.9425866007804871, 'text': '3.0',
              'text_region': [[979.0, 619.0], [1023.0, 635.0], [1013.0, 664.0], [969.0, 648.0]]},
             {'confidence': 0.7618086338043213, 'text': 'mA',
              'text_region': [[703.0, 682.0], [803.0, 676.0], [807.0, 736.0], [707.0, 741.0]]},
             {'confidence': 0.8558327555656433, 'text': '?69C20',
              'text_region': [[505.0, 772.0], [621.0, 779.0], [619.0, 813.0], [503.0, 805.0]]},
             {'confidence': 0.8069597482681274, 'text': '01.51',
              'text_region': [[927.0, 785.0], [1019.0, 780.0], [1021.0, 810.0], [929.0, 814.0]]},
             {'confidence': 0.985348641872406, 'text': 'GB/T7676-98',
              'text_region': [[504.0, 812.0], [634.0, 812.0], [634.0, 840.0], [504.0, 840.0]]},
             {'confidence': 0.9897659420967102, 'text': '1906',
              'text_region': [[634.0, 806.0], [702.0, 806.0], [702.0, 834.0], [634.0, 834.0]]},
             {'confidence': 0.9612194299697876, 'text': 'NH228',
              'text_region': [[834.0, 822.0], [876.0, 822.0], [876.0, 836.0], [834.0, 836.0]]},
             {'confidence': 0.9562859535217285, 'text': '金冠电气股份有限公司',
              'text_region': [[556.0, 923.0], [956.0, 921.0], [956.0, 963.0], [556.0, 965.0]]},
             {'confidence': 0.9187613129615784, 'text': 'P2761T18',
              'text_region': [[202.0, 971.0], [458.0, 971.0], [458.0, 1005.0], [202.0, 1005.0]]},
             {'confidence': 0.8859245181083679, 'text': 'Z0040D19070182预置点',
              'text_region': [[542.0, 967.0], [1064.0, 967.0], [1064.0, 1014.0], [542.0, 1014.0]]},
             {'confidence': 0.9755998849868774, 'text': '贵',
              'text_region': [[2.0, 1011.0], [80.0, 1011.0], [80.0, 1079.0], [2.0, 1079.0]]}]
        ]
        }
