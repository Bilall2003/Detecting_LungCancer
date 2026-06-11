import streamlit as st
from src.main import func

image=st.file_uploader("Upload image",help="only one accepted at a time")
fig, img, smooth, enhanced, mask, area, perimeter, ecc = func(image)

st.pyplot(fig)

st.subheader("Features Extracted")
st.write("Area:", area)
st.write("Perimeter:", perimeter)
st.write("Eccentricity:", ecc)