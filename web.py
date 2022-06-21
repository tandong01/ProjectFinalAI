# Đoạn code này không cần quan tâm có chạy được trên python hay không, vì nó sẽ chạy trực tiếp trên github, do trên python chưa cài streamlit
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_option_menu import option_menu
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("model.h5") #model m train

### load file
uploaded_file = st.file_uploader("Choose an image file", type=["jpg","jpeg","png"])

classes = ['Kohaku', 'Ginrin', 'Goshiki', 'Hikarimuji', 'Hikarimoyo', 'Kumonryu', 'Kujaku', 'Doitsu', 'Chagoi',
               'Ochiba', 'Taisho Sanke', 'Showa ', 'Utsuri', 'Bekko', 'Asagi', 'Shusui', 'Tancho', 'Goromo']
 
if uploaded_file is not None:
    # Convert the file
    img = image.load_img(uploaded_file,target_size=(224,224)) 
    img = img_to_array(img)
    img = img.reshape(1,224,224,3)
    img = img.astype('float32')
    img = img/255
        
    #Button: nút dự đoán sau khi up ảnh
    Genrate_pred = st.button("Dự đoán") 
    
    if Genrate_pred:
    
        prediction = model.predict(img).argmax()
        st.write("Kết quả dự đoán của hình này là:{}**".format(classes [prediction])) 