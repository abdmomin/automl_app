import streamlit as st
import pandas as pd
from PIL import Image
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import os
import pycaret.classification as clf
import pycaret.regression as reg


image = Image.open('robot_image.png')

with st.sidebar:
    st.image(image=image)
    st.title('Auto ML App')
    choice = st.radio(label='Choose an option', options=['Upload', 'EDA', 'Machine Learning', 'Download'])
    st.info('This application will generate EDA and build an automated ML model from uploaded dataset.')

st.write("""
# Automated Data Analysis and Machine Learning Application 
***
""")

if os.path.exists('temp_df.csv'):
    df = pd.read_csv('temp_df.csv', index_col=None)

if choice == 'Upload':
    st.write(""" ### Upload your dataframe for analysis and modelling """)
    data = st.file_uploader(label='Upload your dataframe in CSV format here')
    if data:
        df = pd.read_csv(data, index_col=None)
        df.to_csv('temp_df.csv', index=False)
        st.dataframe(df)

elif choice == 'EDA':
    st.write(""" ### Exploratory Data Analysis """)
    report = df.profile_report()
    st_profile_report(report)

elif choice == 'Machine Learning':
    st.write(""" ### Train Machine Learning Model """)
    target = st.selectbox(label='Select target column', options=df.columns)
    ml = st.selectbox(label='Select Model Type', options=['Classification', 'Regression'])

    if st.button(label='Train Model'):
        if ml == 'Classification':
            clf.setup(df, target=target)
            clf_setup_df = clf.pull()
            st.info('ML Experiment Logs')
            st.dataframe(clf_setup_df)
            best_clf_model = clf.compare_models()
            clf_compare = clf.pull()
            st.info('Best Performing model')
            st.dataframe(clf_compare)
            st.info('Best Model Parameters')
            st.code(best_clf_model)
            clf.save_model(best_clf_model, 'best_model')

        elif ml == 'Regression':
            reg.setup(df, target=target)
            reg_setup_df = reg.pull()
            st.info('ML Experiment Logs')
            st.dataframe(reg_setup_df)
            best_reg_model = reg.compare_models()
            reg_compare = reg.pull()
            st.info('Best Performing model')
            st.dataframe(reg_compare)
            st.info('Best Model Parameters')
            st.code(best_reg_model)
            reg.save_model(best_reg_model, 'best_model')

elif choice == 'Download':
    st.write(""" ### Download Trained Model 
    """)
    with open('best_model.pkl', 'rb') as f_in:
        st.download_button(label='Download The Model', data=f_in, file_name='trained_model.pkl')


