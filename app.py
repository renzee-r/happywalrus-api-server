import json, argparse, time, base64, cv2
from collections import defaultdict
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

HAZARD_NAME = {
    1: 'Cabinet',
    2: 'Chair',
    3: 'Countertop',
    4: 'Dishwasher',
    5: 'Outlet',
    6: 'Oven',
    7: 'Sofa',
    8: 'Stool',
    9: 'Stove',
    10: 'Utensil'
}

HAZARD_PRIORITY = {
    'Falling Hazards': 0,
    'Struck-By Hazards': 1,
    'Thermal Hazards': 2,
    'Poison Hazards': 3,
    'Electrical Hazards': 4
}

HAZARD_CATEGORY = {
    1: 'Poison Hazards',
    2: 'Falling Hazards',
    3: 'Struck-By Hazards',
    4: 'Thermal Hazards',
    5: 'Electrical Hazards',
    6: 'Thermal Hazards',
    7: 'Falling Hazards',
    8: 'Falling Hazards',
    9: 'Thermal Hazards',
    10: 'Struck-By Hazards'
}

HAZARD_DESCRIPTION = {
    'Falling Hazards': ['Falling is the most common cause of injury for children of all ages. \
        A standing and toddling baby has frequent minor falls, can climb chairs, stools, stairs, sofas, or ladders, from which they can fall and become injured. \
        Any corners or rough edges can also cause injury to your child. \
        They may grab unfixed furniture, chairs or stools and cause them to tip over which can injure them.'],

    'Struck-By Hazards': ['A handle jutting over a stove edge presents a temptation for a child to reach up and pull it down. \
        This will quickly cause the pan and all of its contents to fall on your child, with the potential to cause blunt force injuries and burns. \
        Children can also bang their heads on the edges of countertops, kitchen islands, and tables.'],
    
    'Thermal Hazards': ['Ovens, stovetops, and dishwashers have the potential to cause serious and even fatal injuries. \
        Every day, over 300 children ages 0 to 19 are treated in emergency rooms for burn-related injuries and two children die as a result of being burned. \
        Hot liquids cause two out of three burns in small children.'],

    'Poison Hazards': ['Accidental poisoning is common, especially among toddlers aged between one and three years. \
        According to the American Association of Poison Control Centers, nearly 1 million possible poisonings of children under age 5 were reported in 2017. \
        Children explore their environment as part of their normal, natural development and swallowing a poisonous substance, spilling it on the skin, \
        spraying or splashing it in the eye, or inhaling it can all lead to poisoning. \
        Within a kitchen, be especially wary of cleaning products such as bleaches, dishwasher detergents, oven cleaners, drain cleaners, methylated spirits, \
        and turpentine that many people store beneath their sinks.'],
    
    'Electrical Hazards': ['According to the National Fire Protection Association, approximately 2,400 children suffer from severe shock and burn caused by \
        items being poked into electrical receptacles. Even more worrying, approximately 12 children die from these injuries each year. \
        Moreover, when a toddler sees a dangling cable, his or her first instinct is to pull it. If that cable is connected to a small \
        appliance (e.g., toaster, electric mixer) on the counter above, that appliance can fall and injure the child.']
}

HAZARD_SOLUTION = {
    'Falling Hazards': ['Don\'t leave a baby unattended on or near furniture', 
                    'Place bumpers or guards or safety padding on sharp corners of furniture to protect child when they fall',
                    'Keep babies strapped in when using high chairs and furniture',
                    'Use chair locks that can connect chairs to each other and prevents it being pulled by the child'],

    'Struck-By Hazards': ['Align handles towards the back of the stove so your child cannot reach up and grab them',
                    'Place the heaviest objects on lower shelves',
                    'Keep hot foods, liquids, or grease in containers on kitchen surfaces away from the edge and out of a toddler\'s reach',
                    'Shield countertop edges with soft corner protectors',
                    'Keep stools and chairs away from counters to discourage climbing'],
    
    'Thermal Hazards': ['Use back burners to keep the front ones cool. If you must use front burners, turn the pot handles toward the back',
                    'Always have a fire extinguisher or fire blanket in the kitchen',
                    'Keep hot drinks away from children and never hold a child while you have a hot drink',
                    'Install a stove guard around hot plates and stovetops to protect young children from scalds',
                    'Install burner knob covers and appliance locks or latches on the doors to oven (as well as microwaves, refrigerators, and dishwashers) so that your child can\'t turn on the burners or open the oven',
                    'Ensure ovens are anchored to the wall so that children cannot tip them over when their doors are open'],

    'Poison Hazards': ['Lock and secure cabinets and drawers and consider use of magnetic locks',
                    'Keep cleaning supplies out of a child\'s reach, especially avoid storing them unsecured in lower cabinets (e.g., under the sink)',],
    
    'Electrical Hazards': ['Keep all cables safely out of reach or put appliances away when not in use',
                    'Use duct cord cover to put multiple cables in a sleeve to prevent tangling',
                    'Insert electrical outlet covers/safe plates to prevent your child from inserting a fork, finger, tongue, or other object in outlets',
                    'Engage the “lock” feature for any appliances with a button or switch, if they have one',
                    'Replace your old outlet cover with a baby-safe one']
}

HAZARD_PRODUCT = {
    'Falling Hazards': [['Corner Guards', 'https://www.amazon.com/gp/search?ie=UTF8&tag=nmohan-20&linkCode=ur2&linkId=10eb833dc79c986c84712c0af07f7c71&camp=1789&creative=9325&index=aps&keywords=Baby proof corner guards'],
                    ['Drawer Locks', 'https://www.amazon.com/gp/search?ie=UTF8&tag=nmohan-20&linkCode=ur2&linkId=c047a8de2c4960e814436ce9663a0ea9&camp=1789&creative=9325&index=aps&keywords=Child proof drawer locks'],
                    ['Child Safety Gates', 'https://www.amazon.com/gp/search?ie=UTF8&tag=nmohan-20&linkCode=ur2&linkId=23be7c709077a8d6416f9e66f20602db&camp=1789&creative=9325&index=aps&keywords=Child safety gate'],
                    ['Anti-Tip Restraint', 'https://www.amazon.com/gp/search?ie=UTF8&tag=nmohan-20&linkCode=ur2&linkId=7e4abaa44dbe0f9a02698d60a263fcce&camp=1789&creative=9325&index=aps&keywords=Baby Proofing Hangman Anti-Tip Restraint'],
                    ['Chair Locks', 'https://www.amazon.com/gp/search?ie=UTF8&tag=nmohan-20&linkCode=ur2&linkId=1d68d26f396d951e7d6addb1cb756389&camp=1789&creative=9325&index=aps&keywords=child proof chair locks']],
    
    'Struck-By Hazards': [['Corner Guards', 'https://www.amazon.com/gp/search?ie=UTF8&tag=nmohan-20&linkCode=ur2&linkId=10eb833dc79c986c84712c0af07f7c71&camp=1789&creative=9325&index=aps&keywords=Baby proof corner guards'],
                    ['Door Knob Covers', 'https://www.amazon.com/gp/search?ie=UTF8&tag=nmohan-20&linkCode=ur2&linkId=785b629c23df1d2cb1805aa9414350bb&camp=1789&creative=9325&index=aps&keywords=Baby proof door knob cover'],
                    ['Child Safety Gates', 'https://www.amazon.com/gp/search?ie=UTF8&tag=nmohan-20&linkCode=ur2&linkId=23be7c709077a8d6416f9e66f20602db&camp=1789&creative=9325&index=aps&keywords=Child safety gate'],
                    ['Appliance Locks/Latches', 'https://www.amazon.com/gp/search?ie=UTF8&tag=nmohan-20&linkCode=ur2&linkId=d3dd84c17f91b6c438d46e8d8ff4679c&camp=1789&creative=9325&index=aps&keywords=Appliance locks or latches']],
    
    'Thermal Hazards': [['Oven Locks', 'https://www.amazon.com/gp/search?ie=UTF8&tag=nmohan-20&linkCode=ur2&linkId=ab28f613d08ce7f3fb922aa7bd6730aa&camp=1789&creative=9325&index=aps&keywords=Child Proofing oven locks'],
                    ['Stove Knob Covers', 'https://www.amazon.com/gp/search?ie=UTF8&tag=nmohan-20&linkCode=ur2&linkId=d09338e7b50502effbaa912542736137&camp=1789&creative=9325&index=aps&keywords=Child proof stove knob cover'],
                    ['Oven Anchors', 'https://www.amazon.com/gp/search?ie=UTF8&tag=nmohan-20&linkCode=ur2&linkId=25414bb81508eccd758483fb2a0d7c8e&camp=1789&creative=9325&index=aps&keywords=Oven Anti-Tip Bracket']],
    
    'Poison Hazards': [['Cabinet Latches', 'https://www.amazon.com/gp/search?ie=UTF8&tag=nmohan-20&linkCode=ur2&linkId=86e8163554ba1b9ff4a084a9e8dc9d68&camp=1789&creative=9325&index=aps&keywords=Baby proof cabinet latches'],
                    ['Child Safety Locks', 'https://www.amazon.com/gp/search?ie=UTF8&tag=nmohan-20&linkCode=ur2&linkId=daab8bb02059e2fa8adb2bb31966535b&camp=1789&creative=9325&index=aps&keywords=Child safety locks']],

    'Electrical Hazards': [['Outlet Covers', 'https://www.amazon.com/gp/search?ie=UTF8&tag=nmohan-20&linkCode=ur2&linkId=4ddde2580f0aeb4d72fa6c1ef9dde690&camp=1789&creative=9325&index=aps&keywords=Baby proof outlet covers'],
                    ['Power Strip Covers', 'https://www.amazon.com/gp/search?ie=UTF8&tag=nmohan-20&linkCode=ur2&linkId=eccdc7fe32c9e172d499831068577aa6&camp=1789&creative=9325&index=aps&keywords=Baby Proofing Power Strip Cover'],
                    ['Duct Cord Covers', 'https://www.amazon.com/gp/search?ie=UTF8&tag=nmohan-20&linkCode=ur2&linkId=fe4369d3c2510fb9d97918c9af8e6c2d&camp=1789&creative=9325&index=aps&keywords=Duct cord cover']]
}

def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

def load_graph(model_file):
    print('Loading graph 1...', flush=True)
    graph = tf.Graph()
    
    print('Loading graph 2...', flush=True)
    with graph.as_default():
        print('Loading graph 3...', flush=True)
        graph_def = tf.GraphDef()
        print('Loading graph 4...', flush=True)

        with tf.gfile.GFile(model_file, "rb") as f:
            print('Loading graph 5...', flush=True)
            graph_def.ParseFromString(f.read())
            print('Loading graph 6...', flush=True)
            tf.import_graph_def(graph_def, name='')
    
    print('Graph Loaded!', flush=True)
    return graph

##################################################
# API part
##################################################
app = Flask(__name__)
cors = CORS(app)
graph = load_graph('frozen_inference_graph.pb')

@app.route("/predict", methods=['POST'])
def predict():
    print('Starting prediction...', flush=True)
    start = time.time()

    # print('Loading graph 1...', flush=True)
    # graph = tf.Graph()
    # print('Loading graph 2...', flush=True)
    # with graph.as_default():
    #     print('Loading graph 3...', flush=True)
    #     od_graph_def = tf.GraphDef()
    #     print('Loading graph 4...', flush=True)

    #     with tf.gfile.GFile('frozen_inference_graph.pb', 'rb') as fid:
    #         print('Loading graph 5...', flush=True)
    #         serialized_graph = fid.read()
    #         print('Loading graph 6...', flush=True)
    #         od_graph_def.ParseFromString(serialized_graph)
    #         print('Loading graph 7...', flush=True)
    #         tf.import_graph_def(od_graph_def, name='')
    #         print('Loading graph 8...', flush=True)

    sess = tf.Session(graph=graph)

    print('Started session...', flush=True)

    # Define input and output tensors (i.e. data) for the object detection classifier
    # Input tensor is the image
    image_tensor = graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = graph.get_tensor_by_name('detection_scores:0')
    detection_classes = graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = graph.get_tensor_by_name('num_detections:0')

    # print(request.json)
    imageFile = readb64(request.json['file'])
    print('Loaded image...', flush=True)

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    # image = cv2.imread(imageFile)
    image_expanded = np.expand_dims(imageFile, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    print('Performed prediction...', flush=True)

    pred_scores = scores[0]
    pred_boxes = boxes[0]
    pred_classes = classes[0]

    hazard_dict = defaultdict(list)
    for i, score in enumerate(pred_scores):
        if score >= .7:
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
            'priority': HAZARD_PRIORITY[hazard_category],
            'description': HAZARD_DESCRIPTION[hazard_category],
            'solution': HAZARD_SOLUTION[hazard_category],
            'product': HAZARD_PRODUCT[hazard_category],
            'objects': objects
            # # 'score' : pred_score, 
            # 'boxes' : boxes
        })

    hazards = sorted(hazards, key = lambda i: i['priority']) 
    print('Finished post-processing...', flush=True)

    print("Time spent handling the request: %f" % (time.time() - start), flush=True)

    return jsonify(hazards)

@app.route("/test", methods=['GET'])
def test():
    hazards = [
        {
            "category": "Falling Hazard",
            "description": "Your child can climb chairs and injure themselves by falling off.         Any sharp corners or rough edges can also cause injury to your child.         Your child may also grab these chairs, causing them to tip over.",
            "objects": [
            {
                "box": [
                0.6806271076202393,
                0.4436438977718353,
                0.8952561616897583,
                0.594990611076355
                ],
                "id": "3.0",
                "name": "Stool",
                "score": "0.99889565"
            },
            {
                "box": [
                0.6839051842689514,
                0.8282555341720581,
                0.8962242007255554,
                0.9887881875038147
                ],
                "id": "3.0",
                "name": "Stool",
                "score": "0.9984786"
            },
            {
                "box": [
                0.6868025660514832,
                0.6337275505065918,
                0.7726359367370605,
                0.7954311966896057
                ],
                "id": "1.0",
                "name": "Chair",
                "score": "0.9940637"
            }
            ],
            "solution": "Climbing guards & corner guards"
        },
        {
            "category": "Fire Hazard",
            "description": "During and after use, these appliances produce extreme heat that can cause burns and fires.         Be aware of cooking ware on top or within these appliances as well.         Appliances with doors or handles can be tipped over and cause injury.         If these doors are not secured, your child can climb in.",
            "objects": [
            {
                "box": [
                0.5183354616165161,
                0.35441187024116516,
                0.5431620478630066,
                0.49847567081451416
                ],
                "id": "2.0",
                "name": "Stove",
                "score": "0.99867105"
            },
            {
                "box": [
                0.3086070120334625,
                0.07970692217350006,
                0.49102744460105896,
                0.1521143615245819
                ],
                "id": "4.0",
                "name": "Oven",
                "score": "0.97046924"
            },
            {
                "box": [
                0.4906770586967468,
                0.08424205332994461,
                0.8526846766471863,
                0.16121844947338104
                ],
                "id": "4.0",
                "name": "Oven",
                "score": "0.93306625"
            }
            ],
            "solution": "Heat guards, heat shields, & door guards"
        }
        ]

    return jsonify(hazards)

##################################################
# END API part
##################################################

if __name__ == "__main__":
    print('Starting the API')
    app.run()