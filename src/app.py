import streamlit as st
import eda
import prediction
import os

def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="DemandSense AI", layout="wide")
    
    # Inject CSS Global
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_css(os.path.join(current_dir, "style.css"))

    st.sidebar.markdown("<h1 class='animated-sidebar-text'> DemandSense</h1>", unsafe_allow_html=True)
    st.sidebar.write("**Practitioner:** by Risyadhana Syaifuddin")
    st.sidebar.write("**Project ID:** DemandSense AI")
    st.sidebar.markdown("---")
    
    # Selectbox dengan animasi dari CSS
    menu = st.sidebar.selectbox("Navigate Menu:", ["EDA Analysis", "Demand Forecasting"])

    if menu == "EDA Analysis":
        eda.run()
    else:
        prediction.run()

if __name__ == "__main__":
    main()