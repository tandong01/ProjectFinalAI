import streamlit as st
import tensorflow as tf
import streamlit as st


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model


with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         # PHÂN LOẠI 18 LOẠI CÁ KOI
         """
         )

file = st.file_uploader("Hãy tải file lên tại đây", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)


def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(image, dsize=(224, 224),    interpolation=cv2.INTER_CUBIC))/255.

    img_reshape = img_resize[np.newaxis, ...]

    prediction = model.predict(img_reshape)

    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    st.write(predictions)
    st.write(score)
    class_names = ['Kohaku', 'Ginrin', 'Goshiki', 'Hikarimuji', 'Hikarimoyo', 'Kumonryu', 'Kujaku', 'Doitsu', 'Chagoi',
                   'Ochiba', 'Taisho Sanke', 'Showa ', 'Utsuri', 'Bekko', 'Asagi', 'Shusui', 'Tancho', 'Goromo']
    st.write("Cá trong hình này thuộc loại  {} với độ chính xác {:.2f} %."
          .format(class_names[np.argmax(score)], 100 * np.max(score))
          )

