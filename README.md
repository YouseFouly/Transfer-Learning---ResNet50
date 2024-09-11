# Transfer-Learning---ResNet50
Transfer Learning for CIFAR-10 Object Recognition using ResNet50

This project demonstrates the application of transfer learning for object recognition on the CIFAR-10 dataset using the ResNet50 architecture. Transfer learning enables leveraging pre-trained models on large datasets (e.g., ImageNet) to solve tasks on smaller datasets, improving accuracy and reducing training time.

- Table of Contents
- Project Overview
- Dataset
- Model Architecture
- Results
- Technologies Used
- Acknowledgments


## Project Overview

The goal of this project is to use transfer learning with the ResNet50 architecture to classify images from the CIFAR-10 dataset, which consists of 10 object categories. By utilizing a pre-trained model, the model can effectively learn and generalize from limited training data.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes

Classes include:


![image](https://github.com/user-attachments/assets/007cd96b-a6de-4b9a-9319-9381be400ad6)





* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

## Model Architecture

We used the ResNet50 model, which was pre-trained on the ImageNet dataset. ResNet50 is a deep residual network known for its excellent performance on image classification tasks.

- Transfer Learning: The top layers of ResNet50 were removed, and a new fully connected layer was added to output predictions for 10 classes in CIFAR-10.
- Fine-tuning: The lower layers of ResNet50 were frozen, and only the top layers were trained on CIFAR-10 to prevent overfitting.


![image](https://github.com/user-attachments/assets/ea30edb2-c6c0-4813-bf9d-1a28f0c73aac)



## Results

After training, the model achieved high accuracy on the CIFAR-10 test dataset, demonstrating the effectiveness of transfer learning with a pre-trained ResNet50 model.

* Training Accuracy: 98%
* Test Accuracy: 94%
* Loss: Low validation and training loss values



![Capture555](https://github.com/user-attachments/assets/25bd38db-c36d-427b-8238-4f7039acedd4)






![Capture444](https://github.com/user-attachments/assets/8550b428-f782-4305-9b63-ef6b042821c4)



                                                                          

## Technologies Used

Python
* TensorFlow / Keras
* ResNet50 (pre-trained on ImageNet)
* CIFAR-10 Dataset
* Matplotlib for visualizations
* Jupyter Notebook for code execution\

## Acknowledgments

I would like to express my sincere gratitude to Siddhardhan for his excellent tutorial on YouTube, titled “DL Project 4. CIFAR - 10 Object Recognition using ResNet50 | Deep Learning Projects in Python
”. This video provided invaluable guidance on implementing transfer learning techniques and helped me successfully complete this project. His clear explanations and hands-on approach made complex concepts accessible and practical.
