#!/usr/bin/env python
import os
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    # parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=360)

    parser.add_argument("--model",
                        type=str,
                        default='u2net.onnx',
                        )
    parser.add_argument("--input_size", type=int, default=224)

    args = parser.parse_args()

    return args


def run_inference(onnx_session, input_size, image):
    temp_image = copy.deepcopy(image)
    resize_image = cv.resize(temp_image, dsize=(input_size, input_size))
    x = cv.cvtColor(resize_image, cv.COLOR_BGR2RGB)

    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.astype('float32')
    
    x = x.transpose(2, 0, 1).astype('float32')
    x = x.reshape(-1, 3, input_size, input_size)
    # print(x.shape)
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})
    onnx_result = np.array(onnx_result).squeeze()

    onnx_result = onnx_result.astype('uint8')
    print(onnx_result)

    return onnx_result


def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height


    model_path = args.model
    input_size = args.input_size

     # providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'], ...)
    
    onnx_session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

    elapsed_time = 0.0

    image = cv.imread('test.PNG')

    start_time = time.time()

    onnx_result = run_inference(
        onnx_session,
        input_size,
        image,
    )
    elapsed_time = time.time() - start_time

    elapsed_time_text = "Elapsed time: "
    elapsed_time_text += str(round((elapsed_time * 1000), 1))
    elapsed_time_text += 'ms'
    cv.putText(image, elapsed_time_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               0.7, (0, 255, 0), 1, cv.LINE_AA)
    print(elapsed_time_text)
    debug_image = cv.resize(onnx_result,
                            dsize=(image.shape[1], image.shape[0]))

    cv.imwrite('test.jpg', debug_image)

if __name__ == '__main__':
    main()