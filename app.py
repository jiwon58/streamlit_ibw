import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import AsylumPy as ap
import os
from igor import binarywave as bw

def convert(uploaded_file):
    _, path = tempfile.mkstemp(suffix='.ibw')
    with open(path, 'wb+') as f:
        f.write(uploaded_file.read())
        obj = ap.Image(path)
    return obj

def get_key(channels, val):
    for key, value in channels.items():
        if val == value:
            return key 
    return "key doesn't exist"

def draw_image(tab, data):
    with tab:
        fig, ax = plt.subplots()
        ax.imshow(data, origin='lower')
        st.pyplot(fig)
st.set_page_config(page_title='IBW Viewer', page_icon="🖼", layout='centered',
                   menu_items={
         'About': "Converting IBW without Asylum Software"
     })        
st.header('Igor Binary Converter')

uploaded_files = st.file_uploader("Please Uploade Your Igor Binary Files...", accept_multiple_files=True, type='ibw')

if len(uploaded_files) > 0:

    converted_files = {f.name : convert(f) for f in uploaded_files}
    if len(converted_files.keys()) == 1:
        selected = list(converted_files.keys())[0]
    else:
        selected = st.selectbox('Uploaded Items', converted_files.keys())
    st.write('You selected: ', selected)
    obj = converted_files[selected]
    channels = obj.channels
    selected_channels = st.multiselect('Select channels', channels.values(), default=list(channels.values())[0])
    # st.write(converted_files[selected].channels)
    col1, col2= st.columns([1, 2], gap="small")

    with col1:
        st.write(
        pd.DataFrame.from_dict(
            converted_files[selected].channels, orient='index', columns=['Channel_Name']
            ),
        
        )
    with col2:

        tabs = st.tabs(selected_channels)
        for i in range(len(tabs)):
            channel = get_key(channels, selected_channels[i])
            draw_image(tabs[i], obj[channel])
