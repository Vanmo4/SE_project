import io
from PIL import Image
import numpy as np
import pytest

# Тестирование загрузки модели
def test_load_model():
    import tensorflow as tf
    import streamlit as st
    from tensorflow.keras.applications.efficientnet import EfficientNetB0
    from model_2 import load_model
    
# Тестирование предобработки изображения
def test_preprocess_image():
    from PIL import Image
    from model_2 import preprocess_image
    
    img = Image.new('RGB', (224, 224))
    preprocessed_img = preprocess_image(img)
    
    assert preprocessed_img is not None
    assert preprocessed_img.shape == (1, 224, 224, 3)



if __name__ == "__main__":
    pytest.main(["-s"])
