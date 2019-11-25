import json, argparse, time, base64, cv2
from collections import defaultdict
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

HAZARD_NAME = {
    1: 'Chair',
    2: 'Stove',
    3: 'Stool',
    4: 'Oven',
    5: 'Ladder',
    6: 'Sofa',
    7: 'Stove/Oven'
}

HAZARD_CATEGORY = {
    1: 'Falling Hazard',
    2: 'Fire Hazard',
    3: 'Falling Hazard',
    4: 'Fire Hazard',
    5: 'Falling Hazard',
    6: 'Falling Hazard',
    7: 'Fire Hazard'
}

HAZARD_DESCRIPTION = {
    'Falling Hazard': 'Your child can climb chairs and injure themselves by falling off. \
        Any sharp corners or rough edges can also cause injury to your child. \
        Your child may also grab these chairs, causing them to tip over.',
    'Fire Hazard': 'During and after use, these appliances produce extreme heat that can cause burns and fires. \
        Be aware of cooking ware on top or within these appliances as well. \
        Appliances with doors or handles can be tipped over and cause injury. \
        If these doors are not secured, your child can climb in.'
}

HAZARD_SOLUTION = {
    'Fire Hazard': 'Heat guards, heat shields, & door guards',
    'Falling Hazard': 'Climbing guards & corner guards'
}

def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

##################################################
# API part
##################################################
app = Flask(__name__)
cors = CORS(app)
@app.route("/predict", methods=['POST'])
def predict():
    start = time.time()

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile('gs://happy-walrus.appspot.com/frozen_inference_graph.pb', 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # print(request.json)
    imageFile = readb64(request.json['file'])

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    # image = cv2.imread(imageFile)
    image_expanded = np.expand_dims(imageFile, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    pred_scores = scores[0]
    pred_boxes = boxes[0]
    pred_classes = classes[0]

    hazard_dict = defaultdict(list)
    for i, score in enumerate(pred_scores):
        if score >= .6:
            hazard_object = {
                'score': str(score),
                'box': pred_boxes[i].tolist(),
                'id': str(pred_classes[i]),
                'name': HAZARD_NAME[pred_classes[i]]
            }

            hazard_dict[HAZARD_CATEGORY[pred_classes[i]]].append(hazard_object)

    hazards = []
    for hazard_category, objects in hazard_dict.items():
        hazards.append({
            'category': hazard_category,
            'description': HAZARD_DESCRIPTION[hazard_category],
            'solution': HAZARD_SOLUTION[hazard_category],
            'objects': objects
            # # 'score' : pred_score, 
            # 'boxes' : boxes
        })

    print("Time spent handling the request: %f" % (time.time() - start))

    return jsonify(hazards)

##################################################
# END API part
##################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="inference_graph/frozen_inference_graph.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    ##################################################
    # Tensorflow part
    ##################################################
    print('Loading the model')
    global detection_graph, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections

    # detection_graph = tf.Graph()
    # with detection_graph.as_default():
    #     od_graph_def = tf.GraphDef()
    #     with tf.gfile.GFile(args.frozen_model_filename, 'rb') as fid:
    #         serialized_graph = fid.read()
    #         od_graph_def.ParseFromString(serialized_graph)
    #         tf.import_graph_def(od_graph_def, name='')

    #     sess = tf.Session(graph=detection_graph)

    # # Define input and output tensors (i.e. data) for the object detection classifier
    # # Input tensor is the image
    # image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # # Output tensors are the detection boxes, scores, and classes
    # # Each box represents a part of the image where a particular object was detected
    # detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # # Each score represents level of confidence for each of the objects.
    # # The score is shown on the result image, together with the class label.
    # detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    # detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # # Number of objects detected
    # num_detections = detection_graph.get_tensor_by_name('num_detections:0')


    ##################################################
    # END Tensorflow part
    ##################################################

    print('Starting the API')
    app.run()