import streamlit as st
from src.main import func
import numpy as np
import pandas as pd


fig,img=func()
st.subheader("Original image ")

st.pyplot(fig)
st.write("Image shape:", img.shape)
st.write("Pixel range:", img.min(), "to", img.max())