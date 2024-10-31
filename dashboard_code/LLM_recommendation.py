import streamlit as st
from streamlit_searchbox import st_searchbox

options = ["Apple", "Banana", "Cherry", "Date", "Elderberry", "Fig", "Grape", "Honeydew", "Kiwi", "Lemon"]

selected_option = st_searchbox("Search for a fruit:", options)

if selected_option:
    st.write(f"You selected: {selected_option}")
else:
    st.write("Please select a fruit.")


