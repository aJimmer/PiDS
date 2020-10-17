import tflite_runtime.interpreter as tflite
import argparse
import time
import pandas as pd
import numpy as np

#interpreter = tflite.Interpreter(model_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])

def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    return tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[
            tflite.load_delegate('libedgetpu.so.1', {'device': device[0]} if device else {})])

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

    #image = Image.open(args.input)
    data = pd.read_csv(args.input)
    data_id = data.ts
    data = data.drop('ts', axis=1)
    #input_data = np.array(data.values, dtype=np.float32)
    #output_details = interpreter.get_output_details()
 
    #scale = detect.set_input(interpreter, image.size,lambda size: image.resize(size, Image.ANTIALIAS))


    print('----INFERENCE TIME----')
    print('Note: The first inference is slow because it includes', 'loading the model into Edge TPU memory.')

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    
    print("Success!", input_shape, data.values)

    for i in range(data.shape[0]):
        input_data = np.array(data.values[i], dtype=np.float32)

        #interpreter.set_tensor(input_details[0]['index'], input_data)

        #interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        #output_data = interpreter.get_tensor(output_details[0]['index'])
        #print(output_data)
'''
    for _ in range(args.count):
        start = time.perf_counter()
        interpreter.set_tensor(input_data)
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)
        print('%.2f ms' % (inference_time * 1000))

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