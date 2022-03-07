import os
import numpy as np
import pandas as pd
import pickle
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import h5py
from joblib import Parallel, delayed

if __name__ == "__main__":
    train_test_split_file = "/Users/nduginets/PycharmProjects/ISIC2018/data/train_test_id.pickle"
    with open(train_test_split_file, 'rb') as f:
        train_test_id = pickle.load(f)

    train_subset = train_test_id.loc[train_test_id["Split"] == "train"]
    test_subset = train_test_id.loc[train_test_id["Split"] == "test"]
    fake_train = []
    for idx, row in train_subset.iterrows():
        df = row.to_frame().T
        df.at[df.index[0], "Class"] = "fake"
        id_name = df.at[df.index[0], "ID"]
        index = "1" + id_name.split("_")[1][1:]
        df.at[df.index[0], "ID"] = "ISIC_" + index


        fake_train.append(df)
    frame = pd.concat([train_subset, pd.concat(fake_train), test_subset], ignore_index=True)
    frame.to_pickle("/Users/nduginets/PycharmProjects/ISIC2018/data/fake_train_test_id.pickle")
