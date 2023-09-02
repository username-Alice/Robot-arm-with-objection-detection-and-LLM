'''another llm approach, compare the similarity between user's prompt and object_list+object's description'''
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)
# Define a list of objects and their descriptions
object_list = {
    'person': 'a human being',
    'bicycle': 'a vehicle with two wheels',
    'car': 'a vehicle with four wheels that is powered by an engine',
    'motorcycle': 'a vehicle with two wheels and an engine',
    'airplane': 'a vehicle that can fly through the air',
    'bus': 'a large vehicle that is designed to carry passengers',
    'train': 'a long vehicle that runs on tracks and is used for transporting people or goods',
    'truck': 'a large vehicle that is used for transporting goods',
    'boat': 'a vehicle that is used for traveling on water',
    'traffic light': 'a device that is used to control the flow of traffic',
    'fire hydrant': 'a device that is used to supply water for fighting fires',
    'stop sign': 'a sign that is used to indicate that vehicles should stop',
    'parking meter': 'a device that is used to collect money for parking',
    'bench': 'a long seat for two or more people, typically made of wood or metal',
    'bird': 'a warm-blooded vertebrate with feathers and wings',
    'cat': 'a small carnivorous mammal with soft fur and retractable claws',
    'dog': 'a domesticated carnivorous mammal with a snout and claws',
    'horse': 'a large mammal with four legs and a mane, used for riding or racing',
    'sheep': 'a domesticated ruminant mammal with wool and horns',
    'cow': 'a domesticated mammal that is raised for meat or milk',
    'elephant': 'a large mammal with a long nose and tusks, found in Africa and Asia',
    'bear': 'a large carnivorous mammal with shaggy fur and a short tail',
    'zebra': 'an African mammal with black-and-white stripes',
    'giraffe': 'a large African mammal with a very long neck and legs',
    'backpack': 'a bag that is carried on the back',
    'umbrella': 'a device that is used to protect against rain or sun',
    'handbag': 'a bag that is carried by hand or over the shoulder',
    'tie': 'a long piece of cloth that is worn around the neck with a knot in front',
    'suitcase': 'a rectangular case with a handle that is used for carrying clothes and other items',
    'frisbee': 'a plastic disc that is thrown for recreation or sport',
    'skis': 'long narrow strips of wood or metal that are worn on the feet for gliding over snow',
    'snowboard': 'a flat board that is used for gliding over snow',
    'sports ball': 'a ball that is used in various sports, such as soccer, basketball, or football',
    'kite': 'a toy that is flown in the air by a string',
    'baseball bat': 'a wooden or metal club that is used for hitting a baseball',
    'baseball glove': 'a leather glove that is worn by a baseball player for catching the ball',
    'skateboard': 'a flat board with four wheels that is used for riding on a hard surface',
    'surfboard': 'a long narrow board that is used for riding waves in the ocean',
    'tennis racket': 'a device that is used for hitting a ball in the game of tennis',
    'bottle': 'a container with a narrow neck that is used for holding liquids',
    'wine glass': 'a glass that is used for drinking wine',
    'cup': 'a small open container that is used for drinking or holding liquids',
    'fork': 'a utensil with two or more prongs that is used for eating or serving food',
    'knife': 'a utensil with a sharp blade that is used for cutting or slicing food',
    'spoon': 'a utensil with a shallow bowl and a handle that is used for eating or serving food',
    'bowl': 'a round or oval-shaped container that is used for holding food or liquid',
    'banana': 'a long curved fruit with a yellow or green skin and a soft inside',
    'apple': 'a round fruit with red or green skin and a white inside',
    'sandwich': 'a food item consisting of one or more types of food, such as meat or cheese, placed on or between slices of bread',
    'orange': 'a round citrus fruit with a thick orange skin and a juicy inside',
    'broccoli': 'a green vegetable with small flower buds and thick stalks',
    'carrot': 'a long orange root vegetable',
    'hot dog': 'a cooked sausage that is served in along bun and often topped with condiments',
    'pizza': 'a flat bread that is topped with tomato sauce, cheese, and various toppings',
    'donut': 'a type of fried dough that is shaped like a ring or ball and often coated with sugar or icing',
    'cake': 'a sweet baked dessert that is typically made with flour, sugar, eggs, and butter or oil',
    'chair': 'a piece of furniture with a seat, back, and sometimes armrests',
    'couch': 'a long upholstered piece of furniture designed for sitting or reclining',
    'potted plant': 'a plant that is grown in a container instead of in the ground',
    'bed': 'a piece of furniture used for sleeping',
    'dining table': 'a table that is used for eating meals',
    'toilet': 'a fixture used for the disposal of human waste',
    'tv': 'a device used for receiving and displaying television signals',
    'laptop': 'a portable computer that is designed to be used on a person’s lap',
    'mouse': 'a pointing device used to control the movement of a cursor on a computer screen',
    'remote': 'a handheld device used to control electronic devices from a distance',
    'keyboard': 'a device with a set of keys used for typing on a computer or typewriter',
    'cell phone': 'a mobile phone that is used for communication over long distances',
    'microwave': 'a device used for cooking or heating food using microwaves',
    'oven': 'a device used for cooking or baking food',
    'toaster': 'a device used for toasting bread',
    'sink': 'a fixture used for washing hands, dishes, or other objects',
    'refrigerator': 'a device used for storing food and keeping it cool',
    'book': 'a written or printed work that is bound together and has a cover',
    'clock': 'a device used for measuring and displaying time',
    'vase': 'a container used for holding flowers or other decorative objects',
    'scissors': 'a tool used for cutting various materials',
    'teddy bear': 'a stuffed toy bear that is usually covered in soft fur or plush material',
    'hair drier': 'a device used for drying hair',
    'toothbrush': 'a brush used for cleaning teeth'
}
# Load the pre-trained USE model
use_model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

# Define a function to compute cosine similarity
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def get_actual_list(object_names):
    global object_list
    actual_object_list = {name : object_list.get(name) for name in object_names}
    return actual_object_list

def analysis(query, object_names):
    global use_model
    # Preprocess the input query
    #query = 'An object that is at least 3 layers and is food'
    if len(object_names) == 0:
        print("WARNING: No object detected")
        return "",""
    
    actual_objects = get_actual_list(object_names)
    #print(actual_objects)
    query = query.lower().strip()

    # Generate a sentence embedding for the query
    query_embedding = use_model([query])[0]

    # Compute the cosine similarity between the query and each object description
    best_match = None
    best_score = 0
    for object_name, object_description in actual_objects.items():
        object_description_match = use_model([object_description])[0]
        score_1 = cosine_similarity(query_embedding, object_description_match)
        object_name_match = use_model([object_name])[0]
        score_2 = cosine_similarity(query_embedding, object_name_match)
        score = score_1*0.4+score_2*0.6
        if score > best_score:
            best_score = score
            best_match = object_name
    return best_match, best_score

#FOR TESTING ONLY
#query = "I want an apple"
#best_m,best_s = analysis(query,['apple', 'oven', 'orange'])
# Print the best match
#print(f'The best match for "{query}" is "{best_m}" with a score of {best_s:.2f}.')