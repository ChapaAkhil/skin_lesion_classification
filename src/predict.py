import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

def predict_image(image_path, model_path='best_model.h5', class_mapping=None):
    image = Image.open(image_path).resize((28, 28))
    img = np.array(image).reshape(-1, 28, 28, 3)

    model = load_model(model_path)
    result = model.predict(img)
    class_idx = np.argmax(result)

    if class_mapping:
        print(f"Predicted class: {class_mapping[class_idx][1]}")
    else:
        print(f"Predicted class index: {class_idx}")
