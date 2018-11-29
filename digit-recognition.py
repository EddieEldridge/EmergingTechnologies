# Import os
import os

# Supress warnings but not errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import numpy
import numpy as np

# Import tensorflow as tf
import tensorflow as tf

# Supress the warnings even more
tf.logging.set_verbosity(tf.logging.ERROR)

# Import matplotlib as plt
import matplotlib.pyplot as plt

# Import the MNIST dataSet from TensorFlow
from tensorflow.examples.tutorials.mnist import input_data

def LinearClassifier():
    print("\n Running Linear Classification....")

    # Load the mnist dataset from TensorFlow
    mnist = input_data.read_data_sets("MNIST_data")
    print("\n Loaded MNIST successfully....")

    # Create a function to easily get our images and labels from the MNIST dataset
    def dataInput(dataset):
        return dataset.images, dataset.labels.astype(np.int32)

    # Create a variable called feature_columns
    # Reshape the with a shape of 28x28 as this represents the pixel dimensions of our images
    # https://www.tensorflow.org/guide/feature_columns
    feature_columns = [tf.feature_column.numeric_column("mnistData", shape=[28, 28])]
    print("\nCreated and reshaped feature columns....")

    # Create our classifier using TensorFlow's LinearClassifier function
    # We give this classifier 10 classes as the there are 10 outputs for our dataset (0..9)
    # I'll discuss this in more detail in the next notebook
    # https://www.tensorflow.org/api_docs/python/tf/estimator/LinearClassifier
    linearClassifier = tf.estimator.LinearClassifier(
                        feature_columns=feature_columns,
                        optimizer='Ftrl',
                        n_classes=10,
                        )
    print("\nCreated classifier....")

    # Combine the training data and labels into one variable
    # https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/numpy_input_fn
    trainingData = tf.estimator.inputs.numpy_input_fn(
            x={"mnistData": dataInput(mnist.train)[0]},
            y=dataInput(mnist.train)[1],
            num_epochs=None,
            batch_size=100,
            shuffle=True
        )
    print("\nCreated training data....")


    # Combine the test images and test labels into one variable
    # https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/numpy_input_fn
    testingData = tf.estimator.inputs.numpy_input_fn(
            x={"mnistData": dataInput(mnist.test)[0]},
            y=dataInput(mnist.test)[1],
            num_epochs=1,
            shuffle=False
        )
    print("\nCreated test data....")

    # Tell TensorFlow to train the classifier with the training set and corresponding labels in steps of 100
    # https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#train
    print("\nTraining classifier....")
    linearClassifier.train(input_fn=trainingData, steps=1000)
    print("\nClassifier trained!")

    # Assign the images from the MNIST Test set to a variable called testingImagees
    testingImages = mnist.test.images 

    # Evaluate the accuracy of our classifier after using the fit function above
    # https://www.tensorflow.org/api_docs/python/tf/contrib/learn/evaluate
    # Print the accuracy of our fit method as a percentage
    percentageAccuracy = (linearClassifier.evaluate(input_fn=testingData)["accuracy"])
    print("\nTest Accuracy: {0:f}%\n".format(percentageAccuracy*100))
    
      # Print menu
    print("\n Would you like to test a specific image with this classifier?")
    print ("""
    1.Yes
    2.No
    """)
    
    # Take input for the above
    lnChoice=input('')

    # Keep menu open while the user wants to test a specific image
    while lnChoice=='1':
        # Prompt the user for the image they wish to test
        imageNum=int(input("\n Please choose an image to test as an integer (1-10000): "))

        # Assign our prediction to a generator object called predictions, passing it in a specific image from our testing array of images
        # Set shuffle to false to ensure we get the right image 
        # https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#predict
        predictions = linearClassifier.predict(input_fn=tf.estimator.inputs.numpy_input_fn(x={"mnistData": np.array([testingImages[imageNum]])}, shuffle=False))


        # Loop through every element of our prediction object
        for p in predictions:
            # Extract just the probability variables from predictions
            probList = (p['probabilities'])
            
            # Get the max value i.e the predicted digit from our list of probabilites
            maxValue = max(probList)

            # Print the predicted value to the screen
            predictedValue = probList.argmax(axis=0)
            # Round the array of probabilities to 2 decimal palces for easier reading
            roundedArray = np.round(probList, decimals=2)

            # Print our predictions
            print("\n------------Prediction------------\n")
            for i in range(0,10):
                print ("Prediction of %i: %.3f" % (i,roundedArray[i]))           
            
            print("\nPrediction accuracy: {0:f}%\n".format(maxValue*100))
            print("Predicted: ", predictedValue)

        # Get the numbers actual value from labels in the test set of images
        actual = dataInput(mnist.test)[1][imageNum]
        print("Actual: ", actual)
        print("\n------------------------------------\n")

        # Prompt the user if they would like to try another image
        print("\n Would you like to test a specific image with this classifier?")
        print ("""
            1.Yes
            2.No
            """)
        lnChoice=input('')

# Function for running our DNN Classification of MNIST      
def DNNClassifier():
    print("\nRunning Deep Neural Network Classification....\n")

    # Load the mnist dataset from TensorFlow
    mnist = input_data.read_data_sets('MNIST_data')
    print("\nLoaded MNIST successfully....")

    # Create a function to easily get our images and labels from the MNIST dataset
    def dataInput(dataset):
        return dataset.images, dataset.labels.astype(np.int32)

    # Create a variable called feature_columns
    # Reshape the with a shape of 28x28 as this represents the pixel dimensions of our images
    # https://www.tensorflow.org/guide/feature_columns
    feature_columns = [tf.feature_column.numeric_column("mnistData", shape=[28, 28])]
    print("\nCreated and reshaped feature columns....")

    # Create our classifier using TensorFlow's DNNClassifier function
    # We give this classifier 10 classes as the there are 10 outputs for our dataset (0..9)
    # I'll discuss this in more detail in the next notebook
    # https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier
    dnnClassifier = tf.estimator.DNNClassifier(
                        feature_columns=feature_columns,
                        hidden_units=[256, 32],
                        optimizer=tf.train.AdamOptimizer(1e-4),
                        n_classes=10,
                        dropout=0.1,
                        model_dir="./tmp/mnist_model"
                        )
    print("\nCreated classifier....")

     # Combine the training data and labels into one variable
     # https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/numpy_input_fn
    trainingData = tf.estimator.inputs.numpy_input_fn(
            x={"mnistData": dataInput(mnist.train)[0]},
            y=dataInput(mnist.train)[1],
            num_epochs=None,
            batch_size=100,
            shuffle=True
        )
    print("\nCreated training data....")


    # Combine the test images and test labels into one variable
    # https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/numpy_input_fn
    testingData = tf.estimator.inputs.numpy_input_fn(
            x={"mnistData": dataInput(mnist.test)[0]},
            y=dataInput(mnist.test)[1],
            num_epochs=1,
            shuffle=False
        )
    print("\nCreated test data....")

    # Tell TensorFlow to train the classifier with the training set and corresponding labels in steps of 100
    # https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#train
    print("\nTraining classifier....")
    dnnClassifier.train(input_fn=trainingData, steps=1000)
    print("\nClassifier trained!")

    testingImages = mnist.test.images 

    # Print the accuracy of our fit method as a percentage
    # https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#evaluate
    percentageAccuracy = (dnnClassifier.evaluate(input_fn=testingData)["accuracy"])
    print("\nTest Accuracy: {0:f}%\n".format(percentageAccuracy*100))
    
    # Print menu
    print("\n Would you like to test a specific image with this classifier?")
    print ("""
    1.Yes
    2.No
    """)
    
    # Take input for the above
    dnnChoice=input('')

    # Keep menu open while the user wants to test a specific image
    while dnnChoice=='1':
        # Prompt the user for the image they wish to test
        imageNum=int(input("\n Please choose an image to test as an integer (1-10000): "))

        # Assign our prediction to a generator object called predictions, passing it in a specific image from our testing array of images
        # Set shuffle to false to ensure we get the right image 
        # https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#predict
        predictions = dnnClassifier.predict(input_fn=tf.estimator.inputs.numpy_input_fn(x={"mnistData": np.array([testingImages[imageNum]])}, shuffle=False))

        # Loop through every element of our prediction object
        for p in predictions:
            # Extract just the probability variables from predictions
            probList = (p['probabilities'])
            
            # Get the max value i.e the predicted digit from our list of probabilites
            maxValue = max(probList)

            # Print the predicted value to the screen
            predictedValue = probList.argmax(axis=0)
            # Round the array of probabilities to 2 decimal palces for easier reading
            roundedArray = np.round(probList, decimals=2)

            # Print our predictions
            print("\n------------Prediction------------\n")
            for i in range(0,10):
                print ("Prediction of %i: %.3f" % (i,roundedArray[i]))           
            
            print("\nPrediction accuracy: {0:f}%\n".format(maxValue*100))
            print("Predicted: ", predictedValue)

        # Get the numbers actual value from labels in the test set of images
        actual = dataInput(mnist.test)[1][imageNum]
        print("Actual: ", actual)
        print("\n------------------------------------\n")

        # Prompt the user if they would like to try another image
        print("\n Would you like to test a specific image with this classifier?")
        print ("""
            1.Yes
            2.No
            """)
        dnnChoice=input('')

    


# Basic menu
ans=True
while ans:
    print ("""
    ==== MNIST DATASET ====
    1.Run Linear Classification of MNIST
    2.Run Deep Neural Network Classification of MNIST
    3.Exit
    """)
    ans=input("What would you like to do? ") 
    if ans=="1": 
      LinearClassifier()
    elif ans=="2":
      DNNClassifier()
    elif ans=="3":
      print("\n Exiting...")
      exit()
    elif ans !="":
      print("\n Not Valid Choice Try again") 

      

