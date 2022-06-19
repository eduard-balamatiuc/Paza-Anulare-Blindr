# function for face detection with mtcnn
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from numpy import expand_dims
from numpy import asarray
from keras.models import load_model
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


# function that checks if the size is 160x160 and if not, it resizes it
def check_size(image):
    if image.size != (160, 160):
        image = image.resize((160, 160))
    return image


# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = check_size(image)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # initiate the array of faces in photo
    face_array = []
    for result in results:
        # print(result)
        # extract the bounding box from the first face
        x1, y1, width, height = result['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array.append(asarray(image))
    return face_array


def get_embedding(model_location, image_location):
    print('image location', image_location)
    face_pixels = extract_face(image_location)
    model = load_model(model_location)
    yhat = []
    for face_pixel in face_pixels:
        # scale pixel values
        face_pixel = face_pixel.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixel.mean(), face_pixel.std()
        face_pixel = (face_pixel - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixel, axis=0)
        # make prediction to get embedding
        yhat.append(model.predict(samples)[0])
    return yhat


# function that gets the cosine similarity between two vectors
def cosine_similarity(vector_1, list_of_vector_2):
    distances = []
    for vector_2 in list_of_vector_2:
        distances.append(distance.cosine(vector_1, vector_2))
    return distances



# print(cosine_similarity(get_embedding(r'C:\Users\balam\Desktop\FAF_hackathon\facenet_keras.h5', r'C:\Users\balam\Desktop\FAF_hackathon\img\will.jpg'), 
# get_embedding(r'C:\Users\balam\Desktop\FAF_hackathon\facenet_keras.h5', r'C:\Users\balam\Desktop\FAF_hackathon\img\group.jpg'))) 
def check_presence(model_location, image_1_location, image_2_location):
    coefficient = cosine_similarity(get_embedding(model_location, image_1_location), get_embedding(model_location, image_2_location))
    for coef in coefficient:
        if coef < 0.6:
            return True
    return False


