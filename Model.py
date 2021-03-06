import tflite_runtime.interpreter as tflite
import argparse
import time
import pandas as pd
import numpy as np

#interpreter = tflite.Interpreter(model_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])

def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    return tflite.Interpreter(model_path=model_file,
        experimental_delegates=[
           tflite.load_delegate('libedgetpu.so.1', {'device': device[0]} if device else {})])

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=True, help='File path of .tflite file.', default='/home/mendel/PiDS/model.tflite')
    parser.add_argument('-i', '--input', required=True, help='File path of input data.', default='/home/mendel/nms/data/data.csv')
    parser.add_argument('-t', '--threshold', type=float, default=0.38, help='Score threshold for detected objects.')
    parser.add_argument('-o', '--output',help='File path for the result image with annotations')
    parser.add_argument('-c', '--count', type=int, default=5, help='Number of times to run inference')
    args = parser.parse_args()


    data = pd.read_csv(args.input)
    data_id = data.ts
    data = data.drop('ts', axis=1)
    data = data.fillna(0)
    input_data = np.array(data.values, dtype=np.float32)

    print('----INFERENCE TIME----')
    print('Note: The first inference is slow because it includes', 'loading the model into Edge TPU memory.')


    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']


    for i in range(data.shape[0]):
    
        start = time.perf_counter()

        input_tensor = np.expand_dims(input_data[i], axis=0)
        #print('input tensor: ', input_tensor)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)        
        interpreter.invoke()
    
        inference_time = time.perf_counter() - start
        
        output_data = interpreter.get_output_details()
        prediction = interpreter.get_tensor(output_data[0]['index'])[0] 
        mse = np.mean(np.power(input_tensor - prediction, 2), axis=1)
        #score = np.squeeze(interpreter.tensor(interpreter.get_output_details()[2]['index'])())
        #print('mse:', mse)
        #print(output_data, '%.2f ms' % (inference_time * 1000))
        if args.threshold < mse[0]:
            print('%.2f ms' % (inference_time * 1000), mse[0],'anomaly detected!!!')
        else:
            print('%.2f ms' % (inference_time * 1000), mse[0])
        
if __name__ == '__main__':
    main()

'''
Next steps:
    - update interpreter to run on TPU with float 16 values
    - recompile model and optimize for TPU and float 16 values
    - debug Interpreter.invoke with test data
    - commpute reconstruction errors
    - compare reconstruction errors against threshold for normal or malicious 

    Note: on mdt shell startup, python3 Model.py -m /home/mendel/model.tflite -i /home/mendel/nms/data/data.csv can be used to run this code 
    ''' 