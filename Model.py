import tflite_runtime.interpreter as tflite
import argparse
import time
import pandas as pd
import numpy as np

#interpreter = tflite.Interpreter(model_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])

def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    return tflite.Interpreter(model_path=model_file)
       # model_path=model_file,
        #experimental_delegates=[
         #   tflite.load_delegate('libedgetpu.so.1', {'device': device[0]} if device else {})])

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=True, help='File path of .tflite file.', default='/home/mendel/model.tflite')
    parser.add_argument('-i', '--input', required=True, help='File path of input data.', default='/home/mendel/nms/data/data.csv')
    parser.add_argument('-t', '--threshold', type=float, default=0.38, help='Score threshold for detected objects.')
    parser.add_argument('-o', '--output',help='File path for the result image with annotations')
    parser.add_argument('-c', '--count', type=int, default=5, help='Number of times to run inference')
    args = parser.parse_args()

    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    data = pd.read_csv(args.input)
    data_id = data.ts
    data = data.drop('ts', axis=1)
    input_data = np.array(data.values, dtype=np.float32)
 
    #scale = detect.set_input(interpreter, image.size,lambda size: image.resize(size, Image.ANTIALIAS))


    print('----INFERENCE TIME----')
    print('Note: The first inference is slow because it includes', 'loading the model into Edge TPU memory.')

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']

    # Test the model on random input data.
    input_shape = input_details[0]['shape']


    #for i in range(data.shape[0]):
    
    start = time.perf_counter()

    #interpreter.set_tensor(input_details[0]['index'], np.expand_dims(input_data[i], axis=0))
    input_tensor = np.expand_dims(input_data[0], axis=0)
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    
    inference_time = time.perf_counter() - start
    output_data = interpreter.get_output_details()
    prediction = interpreter.get_tensor(output_data[0]['index'])[0] 
    print((prediction).astype(float))
    print(output_data, '%.2f ms' % (inference_time * 1000))
        
'''
    for _ in range(args.count):
        start = time.perf_counter()
        interpreter.set_tensor(input_data)
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)
        

    print('-------RESULTS--------')
    if not objs:
        print('No objects detected')

    for obj in objs:
        print(labels.get(obj.id, obj.id))
        print('  id:    ', obj.id)
        print('  score: ', obj.score)
        print('  bbox:  ', obj.bbox)

    if args.output:
        image = image.convert('RGB')
        draw_objects(ImageDraw.Draw(image), objs, labels)
        image.save(args.output)
        image.show()
        '''
if __name__ == '__main__':
    main()