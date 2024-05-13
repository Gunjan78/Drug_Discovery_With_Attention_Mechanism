import os
import pandas as pd
import tensorflow as tf
import sys
sys.path.append("/")

from drug_molecule_gen.config import config

def load_dataset(file_name):

    file_path = os.path.join(config.DATAPATH,file_name)
    data = pd.read_csv(file_path)

    return data

def save_model(model_to_save):

    saved_file_name = "enc_dec_drug_molecule_gen.keras"
    save_path = os.path.join(config.SAVED_MODEL_PATH,saved_file_name)
    model_to_save.save(save_path)
    print("Model Saved at",save_path)


def load_nn_model(model_to_load):

    save_path = os.path.join(config.SAVED_MODEL_PATH,model_to_load)
    pretrained_model = tf.keras.models.load_model(save_path)
    return pretrained_model


