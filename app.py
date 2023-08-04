import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
    
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

    # plotting
    fig=px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)




