# Import os
import os

# Import gzip to unzip our files
import gzip

# Import cv2 to save the images
import cv2

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

# Supress warnings but not errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def LinearClassifier():
    print("\nRunning Linear Classification....\n")

    # Load the mnist dataset from TensorFlow
    mnist = input_data.read_data_sets("MNIST_data")

    # Check to see if the dataset was found correctly
    if(mnist!=null):
        print("\nLoaded MNIST successfully....\n")
    else:
        print("Please download the dataset first.")
        return

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
    
    # Check to see if the dataset was found correctly
    if(mnist!=null):
        print("\nLoaded MNIST successfully....\n")
    else:
        print("Please download the dataset first.")
        return

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

     # Assign the images from the MNIST Test set to a variable called testingImagees
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

# Function for saving images from MNIST Test set
def ImageSaver():
    # Prompt user for number of images they wish to load
    print("\n Define the range of images you would like to unzip - Range1:Range2 - 1-10000")

    # Get the range from the user
    range1=int(input("\nRange 1: "))
    range2=int(input("\nRange 2: "))

    # Get the number of images
    numImages = range2-range1

    # Define the starting point for the image
    statingPointImages1=(range1*784)+16
    statingPointImages2=statingPointImages1+784

    # Define the starting point for the labels
    statingPointLabels1=range1+8
    statingPointLabels2=statingPointLabels1+1
    
    try:
        # Using gzip we just imported, open the zip files contained in our data folder
        with gzip.open('MNIST_data/t10k-images-idx3-ubyte.gz', 'rb') as file_images:
            image_contents = file_images.read()
            
        # Using gzip we just imported, open the zip files contained in our data folder
        with gzip.open('MNIST_data/t10k-labels-idx1-ubyte.gz', 'rb') as file_labels:
            labels_contents = file_labels.read()
    except:
        print("\nPlease download the files first by using the appropriate function.")
        return

    # Create a directory to store our MNIST images
    dirName='MNIST_images'

    # Check to see if the directory exists before trying to make it
    if(os.path.isdir(dirName)==False):
        os.mkdir(dirName)
        os.chdir(dirName)        
    else:
        os.chdir(dirName)        

    # Loop through the images assigning a corresponding label of the the drawn number
    for x in range(numImages):
            image = ~np.array(list(image_contents[statingPointImages1:statingPointImages2])).reshape(28,28).astype(np.uint8)
            label = np.array(list(labels_contents[statingPointLabels1:statingPointLabels2])).astype(np.uint8)
            
            # Every 784 bytes corresponds to a 1 image so increment by 784
            statingPointImages1+=784
            statingPointImages2+=784

            # Each byte corresponds to a 1 label so increment by 1
            statingPointLabels1+=1
            statingPointLabels2+=1
        
            # Save the images with the following format
            # E.G train-(0)_[7]
            # This means the image is from the test set, is the first image in the set and the drawn image is a 7
            cv2.imwrite('test-(' + str((statingPointLabels1)-9) + ')' + str(label) + '.png', image)

    print("\nYou have successfully unzipped "+str(numImages)+" images to 'MNIST_images'.\n")
    return

# Function to download the MNIST dataset
def DownloadDataset():
    print("\nDownloading dataset...")
     # Load the mnist dataset from TensorFlow
    mnist = input_data.read_data_sets("MNIST_data")
    print("\nDataset downloaded as 'MNIST_data'!\n")

# Basic menu
ans=True
while ans:
    print ("""
    ============ MNIST DATASET ============
    1. Download MNIST dataset via TensorFlow
    2. Unzip and save images from MNIST test set
    3. Run Linear Classification of MNIST dataset
    4. Run Deep Neural Network Classification of MNIST dataset
    5. Exit
    """)
    ans=input("What would you like to do? ") 
    if ans=="1": 
      DownloadDataset()
    elif ans=="2":
      ImageSaver()
    elif ans=="3":
      LinearClassifier()
    elif ans=="4":
      DNNClassifier()
    elif ans=="5":
      print("\n Exiting...")
      exit()
    elif ans !="":
      print("\n Not Valid Choice Try again") 

      

