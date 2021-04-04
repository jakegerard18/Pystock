import streamlit as st
from PIL import Image

def app():
    # Add title
    st.write("""
    # Welcome to Pystock!
    A stock analysis tool built with Python.
    """)

    image = Image.open('./pylogo-1200.png')

    st.image(image)
