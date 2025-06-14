# src/preprocessing.py

import pandas as pd
import numpy as np
import zipfile
import os
import warnings

warnings.filterwarnings('ignore')

def load_and_preprocess_data(zip_path="data/hmnist_28_28_RGB.csv.zip"):
    # Extract if not already extracted
    csv_path = zip_path.replace('.zip', '')
    if not os.path.exists(csv_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(zip_path))

    df = pd.read_csv(csv_path)

    # Shuffle and split
    df = df.sample(frac=1).reset_index(drop=True)
    train_set, test_set = np.array_split(df, [int(0.8 * len(df))])

    x_train = train_set.drop(columns=['label'])
    y_train = train_set['label']
    x_test = test_set.drop(columns=['label'])
    y_test = test_set['label']

    classes = {
        0: ('akiec', 'actinic keratoses and intraepithelial carcinomae'),
        1: ('bcc', 'basal cell carcinoma'),
        2: ('bkl', 'benign keratosis-like lesions'),
        3: ('df', 'dermatofibroma'),
        4: ('nv', 'melanocytic nevi'),
        5: ('vasc', 'pyogenic granulomas and hemorrhage'),
        6: ('mel', 'melanoma'),
    }

    return x_train, y_train, x_test, y_test, classes
