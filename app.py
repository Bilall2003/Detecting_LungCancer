import streamlit as st
from src.main import func

obj=func()
st.subheader("Original image with tumor")
st.pyplot(obj)