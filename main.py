import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from matplotlib import pyplot as plt
import pickle
import gradio as gr
from google.colab import files


class ToxicityModel:
    def __init__(self):
        self.MAX_FEATURES = 200000
        self.model = None
        self.vectorizer = None

    def install_dependencies(self):
        # Install dependencies here
        pass

    def load_data(self):
        files.upload()
        self.df = pd.read_csv("train.csv")

    def preprocess(self):
        X = self.df["comment_text"]
        y = self.df[self.df.columns[2:]].values

        self.vectorizer = TextVectorization(
            max_tokens=self.MAX_FEATURES, output_sequence_length=1800, output_mode="int"
        )
        self.vectorizer.adapt(X.values)
        vectorized_text = self.vectorizer(X.values)

        dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
        dataset = dataset.cache()
        dataset = dataset.shuffle(160000)
        dataset = dataset.batch(16)
        dataset = dataset.prefetch(8)

        train = dataset.take(int(len(dataset) * 0.7))
        val = dataset.skip(int(len(dataset) * 0.7)).take(int(len(dataset) * 0.2))
        test = dataset.skip(int(len(dataset) * 0.9)).take(int(len(dataset) * 0.1))

        self.train = train
        self.val = val
        self.test = test

    def create_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.MAX_FEATURES + 1, 300))
        self.model.add(Bidirectional(LSTM(32, activation="tanh")))
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(6, activation="sigmoid"))

        self.model.compile(loss="BinaryCrossentropy", optimizer="Adam")

    def train_model(self, epochs=1):
        early_stp = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=4)
        mdl_save = ModelCheckpoint(
            "best_model.h5", save_best_only=True, monitor="val_loss", mode="min"
        )
        history = self.model.fit(
            self.train,
            epochs=epochs,
            validation_data=self.val,
            callbacks=[early_stp, mdl_save],
        )

        with open("history.pkl", "wb") as f:
            pickle.dump(history.history, f)

    def make_predictions(self, text):
        input_text = self.vectorizer([text])
        res = self.model.predict(input_text)
        return (res > 0.5).astype(int)

    def evaluate_model(self):
        pre = Precision()
        re = Recall()
        auc = AUC(curve="ROC", multi_label=True, num_labels=6)

        for batch in self.test.as_numpy_iterator():
            X_true, y_true = batch
            yhat = self.model.predict(X_true)

            y_true = y_true.flatten()
            yhat = yhat.flatten()

            pre.update_state(y_true, yhat)
            re.update_state(y_true, yhat)
            auc.update_state(y_true, yhat)

        precision = pre.result().numpy()
        recall = re.result().numpy()
        auc_score = auc.result().numpy()

        print(f"Precision: {precision}, Recall: {recall}, AUC Score: {auc_score}")

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_saved_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)

    def score_comment(self, comment):
        vectorized_comment = self.vectorizer([comment])
        results = self.model.predict(vectorized_comment)

        text = ""
        for idx, col in enumerate(self.df.columns[2:]):
            text += "{}: {}\n".format(col, results[0][idx] > 0.5)

        return text

    def launch_interface(self):
        interface = gr.Interface(
            fn=self.score_comment,
            title="TOXIC COMMENT CLASSIFIER",
            inputs=gr.inputs.Textbox(
                lines=2, placeholder="Comment to score", label="COMMENT"
            ),
            outputs=gr.outputs.Textbox("text", label="OUTPUT"),
            allow_flagging="manual",
            flagging_options=["Incorrect", "Partially Correct", "Correct"],
            flagging_dir="./Flagged data",
        )
        interface.launch(share=True)


toxicity_model = ToxicityModel()
toxicity_model.load_data()
toxicity_model.preprocess()
toxicity_model.create_model()
toxicity_model.train_model(epochs=25)
toxicity_model.evaluate_model()
toxicity_model.save_model("toxicity.h5")

# Load the saved model
toxicity_model.load_saved_model("toxicity.h5")

# Make predictions
predictions = toxicity_model.make_predictions(
    "You freaking suck! I am going to hit you."
)
print(predictions)

# Score a comment using the trained model
comment_score = toxicity_model.score_comment("hey i freaken hate you!")
print(comment_score)

# Launch the Gradio interface
toxicity_model.launch_interface()
