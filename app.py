import streamlit as st 
from fastai.vision.all import *
import pathlib
import plotly as px
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
#title
st.title('Transportni klassifikatsiya qiluvchi model')

st.text("Bu model asosan 3 xil rasmni taniy oladi: Airplane, Car, Boat")

# rasmni joylash
file = st.file_uploader("Rasm yuklang:", type=["png","jpeg","gif","svg"])
if file:
    st.image(image=file)

    # PIL convert
    img = PILImage.create(file)


    # model
    model = load_learner("transport_model.pkl")

    # prediction
    prediction, pred_id, probs= model.predict(img)
    st.success(f"Bashorat: {prediction}")
    st.info(f"Ehtimolligi: {probs[pred_id]*100:.1f}%")




