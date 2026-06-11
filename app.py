import streamlit as st
from src.main import func


obj=func()
st.subheader("Original image ")

st.pyplot(obj)