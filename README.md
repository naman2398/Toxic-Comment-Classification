# Toxic Comment Classification

## Project Description
Toxic comment classification is a crucial task in natural language processing (NLP), aimed at identifying and flagging potentially harmful or offensive comments on online platforms. This project presents a practical implementation of a neural network-based model for toxic comment classification. The model is designed to classify comments into six different categories of toxicity: toxic, severe toxic, obscene, threat, insult, and identity hate.

### Motivation
The motivation behind this project is to combat online toxicity by automating the identification and categorization of toxic comments. By doing so, it aims to create a safer and more respectful online environment.

### What Problem Does It Solve?
This project addresses the problem of identifying and classifying toxic comments in online discussions, making it easier to moderate online communities and protect users from harmful content.

## Key Features
- Data preprocessing to prepare the comment dataset for training.
- A neural network model architecture for comment classification.
- Training and evaluation of the model using precision, recall, and accuracy metrics.
- Making predictions on new comment texts.
- A Gradio-based user interface for testing the model.
- Model deployment as an HDF5 file for easy use.

## Details about the Project

### 1. DataSet

The dataset used for this project is the [Toxic Comment Classification Dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) from Kaggle. It contains a large number of comments, along with labels indicating whether each comment belongs to one or more of the labels. The dataset is split into training and test sets, with the training set containing approximately 160,000 comments and the test set containing approximately 40,000 comments.

### 2. Preprocessing
- The project starts with data preprocessing, including loading the comment dataset from a CSV file.
- Text vectorization is performed using TensorFlow's TextVectorization layer.
- The data is split into training, validation, and testing sets.

### 3. Neural Network Model Architecture
- The model consists of an embedding layer, bidirectional LSTM layer, fully connected layers, and a final dense layer.
- It is compiled with Binary Crossentropy loss and the Adam optimizer.

![image](https://github.com/naman2398/Naman.Portfolio/blob/main/images/toxic.PNG)

### 4. Model Training and Evaluation
- The model is trained on the training dataset for one epoch.
- Evaluation metrics include precision, recall, and categorical accuracy.

### 5. Making Predictions
- The trained model can be used to predict toxicity categories for new comment texts.
- Thresholding is applied to convert probabilities into binary predictions.

### 6. Testing and Deployment with Gradio
- Gradio is used to create a user interface for testing the model.
- The trained model is saved as an HDF5 file (toxicity.h5) for easy deployment.

### Dependencies

To run this project, you will need to have the following dependencies installed:

- Pandas: for data manipulation
- TensorFlow: for deep learning
- Gradio: for creating a user interface
- NumPy: for data handling
- Matplotlib: for data visualization

You can install these dependencies using the following command in a bash terminal:

```bash
pip install pandas tensorflow gradio numpy matplotlib
```


## Getting Started
To use this project, follow these steps:

1. Clone the repository from GitHub.
2. Install the required libraries and dependencies.
3. Run the provided code to train the model.
4. Use the model for comment toxicity prediction via the provided interface.

---

**Note:** For more in-depth information and code examples, please refer to the project's codebase.
