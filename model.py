import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import decode_predictions
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import load_model

model = tf.keras.models.load_model('vgg19.h5')

def preprocess_image(image):
    # make image into numpy array
    image = img_to_array(image)

    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    return image

def prediction(image):
    # predict the probability across all output classes
    predict = model.predict(image)

    # convert the probabilities to class labels
    label = decode_predictions(predict)

    # retrieve the most likely probability
    label = label[0][0]

    # return the classification
    prediction = label[1]
    print('%s (%.2f%%)' % (label[1], label[2]*100))

    return prediction

image = load_img('images/dog.jpg', target_size=(224, 224))
image = preprocess_image(image)
prediction = prediction(image)
print(prediction)
