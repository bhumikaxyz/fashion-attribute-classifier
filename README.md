# Fashion Attribute Classifier

## Overview

The **Fashion Attribute Classifier** is a machine learning web application that uses deep learning to predict fashion product attributes, including  **article type** ,  **color**,  **gender**, and  **season**, based on an uploaded image. The model is built using **VGG16** as the base model and fine-tuned for classification tasks. It leverages a pre-trained Keras model and label encoders to transform model outputs into readable categories.

## Features

* **Article Type Classification** : Predict the type of fashion item (e.g., t-shirt, pants, etc.).
* **Base Color Classification** : Predict the color of the fashion item.
* **Gender Classification** : Classify whether the product is for men or women or unisex.
* **Season Classification** : Predict the season suitable for the fashion item (e.g., winter, summer).

## Dataset

The project uses a **[fashion product dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)** with labeled images, including attributes for article type, color, gender, and season. The model is trained on these attributes and uses the VGG16 architecture as a base.

The saved and serialised model and label encoders can be found [here](https://drive.google.com/drive/folders/1vqXL72OrawtY5MhbsDGYWrLAvodUrVeR?usp=sharing).

## Installation

1. Clone the repository:

   ```python
   git clone https://github.com/bhumikaxyz/fashion-attribute-classifier.git
   ```
2. Navigate to the project directory:

   ```python
   cd fashion-attribute-classifier
   ```
3. Install the required libraries:

   ```python
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit app** :
   Launch the app by running the following command:

   ```python
   streamlit run app.py
   ```
2. **Upload an Image** :

* Upload a fashion product image in `.jpg`, `.jpeg`, or `.png` format.

3. **View Predictions** :

* After uploading the image and clicking "Submit," the app will display the predicted attributes.
