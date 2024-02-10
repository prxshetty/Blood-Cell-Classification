# Blood Cell Classification using MobileNetV2

This project uses transfer learning with MobileNetV2 to classify blood cell images into different types like red blood cells, white blood cells etc.

## Dataset
The dataset used is from Kaggle ([https://www.kaggle.com/datasets/paultimothymooney/blood-cells]) It contains 12,500 augmented images of blood cells across 4 classes:
- Eosinophil
- Lymphocyte
- Monocyte
- Neutrophil
  
## Requirements
Key libaries used:
- TensorFlow 2.x
- Matplotlib
- scikit-learn
- CUDA (Nvidia Graphics Required)

## Preprocessing
The images are resized to 224x224 and pixel values are scaled between -1 and 1 before passing to MobileNetV2 model as the MobileNetV2 model was trained on that specific data.

## Model
A pretrained MobileNetV2 model is used and the top classification layer is replaced for our 4 blood cell type classes. The model is then trained on the training images and evaluated on test set.

## Training
The model is trained using categorical cross entropy loss and Adam optimizer. Callbacks are used for early stopping, model checkpointing and reducing learning rate on plateau. The images are split into training and test sets with training having validation split as well.


## Evaluation
Coming soon...

## Upcoming...
- Use data from National Library of Medicine
- Ill try newer architectures like EfficientNets
- Experiment with somed hyperparameter tuning

