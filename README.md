# Cifar-10 Image Classification in TensorFlow

## Prerequisites

1. Image Classification
2. Convolutional Neutal Networks including basic pooling, convolutional layers with normalization in neural networks and dropout.
3. Data Augmentation
4. Neural Networks
5. NumPy Arrays

## Stepwise Implementation

1. The first step towards writing any code is to import all the required libraries and modules. This includes importing tensorflow and other modules like numpy. If the module is not present then we can download it using "pip install tensorflow" on the cmd. If in Jupyter Notebook we simply type "!pip install tensorflow" in the cell and run it in order to downlaod the module. Other modules can aslo be imported similarly.

![Screenshot 2024-12-11 110355](https://github.com/user-attachments/assets/4c1289b3-73b7-4bc9-b0ca-da8782ca791c)

The output of the above code should display the version of tensorflow you are using(2.17.1) or any other.

2. Now we have the required module support so let's load in our data. The dataset CIFAR-10 is available on tensorflow keras API, and we can downlaod it in our local machine using "tensorflow.keras.datasets.cifar10" and the  distribute it to train and test set using "load_data()" function.

![Screenshot 2024-12-11 110931](https://github.com/user-attachments/assets/517d7c3a-8812-4b2b-b9a9-4a91acb99501)

Herewe can see we have 50000 training images and 10000 test images as specified above and all the images are of 32 by 32 size have 3 color channels i.e. images are color imagws. As well as it is also visible that tehre is only a single label assigned with each image.

3. Until now, we have our data with us. But still, we cannot send it directly to our neural network. We need to process the data in order to sen it to the network. The first thing in the process is to reduce the puxel values. Currently, all the image pixels are in a range of 1-256, and we need to reduce those values to a value ranging between 0 and 1. This enables our model to easily track trends and efficient training. We can do this by dividing all pixel value of 255.0

Another thingh we want top do is to flatten (in simple words arrange them in form of a row) the label values using the flatten() function.

![Screenshot 2024-12-11 111545](https://github.com/user-attachments/assets/75c2f9ea-51e1-4813-bc0c-ab5b3e792efd)

4. Now this is a good rime to  see few images of our dataset. we can visualize it in a subplot grid form. Since the image size is just 32x32 so don't exoect much from the image. It would be a blurred one. We can do the visualization using the subplot() function from matplotlib and looping over the first 25 images from our training dataset portion.

![image](https://github.com/user-attachments/assets/3b3b834a-7f94-422e-be03-a946ba356f15)

Though the images are not clear there are enough pixels for us to specify which object is there in those images

5. After completing all the steps now is the time to build our model. We are going to use a CNN to traun our model. It includes using a convolutional layer in this which is Conv2d layer as well as pooling and normalization metyhods. Finally, we'll pass it into a dense layer and the final layer will be our output layer. We are using 'relu' activation function. The output lyer uses a "softmax" function.

6. Our model is now ready, it's time to compile it. We rae using mofrl.compile() function to compile our model. For the parameters, we are using:

a. adam optimizer

b. sparse_caategorical_crossentropy

c. metrics=['accuracy']

![Screenshot 2024-12-11 114217](https://github.com/user-attachments/assets/9924dfe0-f585-43b7-9d6e-8327c15622ea)

7. Now let's fit our model using model.fit() passing all our data to it. We are going to train our model till 50 epochs, it gives us a fair result though you can  change it if you want.

![Screenshot 2024-12-11 114420](https://github.com/user-attachments/assets/5e322503-4333-45b6-845b-82d74b85014c)

8. After this model is trained, though it will work fine but to make our model much more accurate we can add data augmentation on our data and then train it again. Calling model.fit() afain on augmented data will continue training where it left off. We are going to train our data on a batch size of 32 and we are going to shift the range of width and height by 0.1 and flip the images horizontally. Rhen call model.fit() again for 50 epochs.

![Screenshot 2024-12-11 114715](https://github.com/user-attachments/assets/2be9f79d-131e-4146-9875-ab8ec3a160a6)

![Screenshot 2024-12-11 114729](https://github.com/user-attachments/assets/8332a51c-bd2b-4dcd-b69a-a5bf874fcc61)

9. Now we have trained our model. Before making any predictions from it let's visualize the accuracy per iteration for better analysis. Though there are other methofd that include confusion matrix for better analysis of the model.

![Screenshot 2024-12-11 114924](https://github.com/user-attachments/assets/78476c86-d16b-40dc-9af1-6b3f40b2c658)

Let's make a prediction over an image from our model using model.predict() function. Before sending the image to our model we need to again reduce the pixel values between 0 and 1 and change it's shape to (1,32,32,3) as our model expects the input to be in this form only. To make things easy ;et us take an image from the dataset itself. It is already in reduced pixels format still we have to reshape it (1,32,32,3) using reshape() function. Since we are uding the data from the dataset we can compare the predicted output and original output.

![Screenshot 2024-12-11 115018](https://github.com/user-attachments/assets/93fbe628-0a78-417d-b08c-cf3b19bc7a8d)

Finally, let's save our model using model.save() function as h5 file.

![Screenshot 2024-12-11 115241](https://github.com/user-attachments/assets/d96f04ae-1a76-4dcd-955b-123d20f4f037)







