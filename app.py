import streamlit as st
from src.main import func

image_path = "asset/LungCancerCTscan.jpg"

fig, img, smooth, enhanced, mask, area, perimeter, ecc = func(image_path)

st.pyplot(fig)

st.subheader("Extracted Features")

st.write("Area", area)
st.write("Perimeter", round(perimeter, 2))
st.write("Eccentricity", round(ecc, 4))