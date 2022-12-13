# Toxic-Comment-Classification



## Problem Statement

Toxic comments are a major problem on the internet, leading to negative experiences for both individuals and communities. The goal of this project is to build a model that can classify toxic comments so that they can be flagged and removed.
To be specific " build a multi-headed model that’s capable of detecting different types of toxicity like threats, obscenity, insults, and identity-based hate".

## Data

The dataset used for this project is the [Toxic Comment Classification Dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) from Kaggle. It contains a large number of comments, along with labels indicating whether each comment belongs to one or more of the labels. The dataset is split into training and test sets, with the training set containing approximately 160,000 comments and the test set containing approximately 40,000 comments.

## Evaluation

The model will be evaluated using the [ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) metric. AUC scores can range from 0 to 1, with a higher score indicating a better model.
