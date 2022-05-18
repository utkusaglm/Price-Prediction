from cgitb import text
from distutils.command.upload import upload
import streamlit as st
from serve_model import insert_new_data,train_model_and_save,serve_model
from save_and_train_model import make_data_ready
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import pickle
import base64
# from utils import (smiles_to_mol, mol_file_to_mol, 
#                    draw_molecule, mol_to_tensor_graph, get_model_predictions)

# ----------- General things
st.title('Stock Market Analysis Using ML')
valid_molecule = True
loaded_molecule = None
selection = None
submit = None

# ----------- Sidebar
page = st.sidebar.selectbox('Page Navigation', ["Random_F", "Logistic_R"],key='page_nav')

# st.sidebar.markdown("""---""")
# st.sidebar.write("Created by [DeepFindr](https://www.youtube.com/channel/UCScjF2g0_ZNy0Yv3KbsbR7Q)")
# st.sidebar.image("assets/logo.png", width=100)

def main():
    if st.session_state.page_nav =='Random_F':
        st.text_input('Train, Test or Download The Model (Train, Test, Download)',key='trained_model')
        if st.session_state.trained_model =='Train':
            st.text_input('Write one of them (MSFT, AAPL, NVDA, UBER) to see score',key='which_model')
            if st.session_state.which_model in ['MSFT', 'AAPL', 'NVDA', 'UBER'] :
                st.write(st.session_state.which_model)
                insert_new_data()
                df,r_f,l_r = train_model_and_save()
                st.markdown('Latest Data')
                st.dataframe(df.head())
                st.write('Train Finished')
                accuracy_train = st.write(f" accuracy_train: { r_f[st.session_state.which_model ][0] }")
                accuracy_test = st.write(f" accuracy_test: { r_f[st.session_state.which_model ][1] }") 
                roc_auc = st.write(f" roc_auc: { r_f[st.session_state.which_model ][2] }")
                fig, ax = plt.subplots()
                ax = plt.plot(r_f[st.session_state.which_model ][3],r_f[st.session_state.which_model ][4])
                st.pyplot(fig)
        elif st.session_state.trained_model =='Test':
            st.file_uploader(label='Try model with the data your own (MSFT, AAPL, NVDA, UBER) ',type='csv',key='test_data')
            d_f_e = pd.read_csv('MSFT.csv')
            st.write('Example input data')
            st.dataframe(d_f_e.head())
            st.text_input('Write one of them (MSFT, AAPL, NVDA, UBER) to see score',key='which_model_test')
            if  st.session_state.which_model_test in ['MSFT', 'AAPL', 'NVDA', 'UBER'] :
                st.write(st.session_state.test_data)
                df,t_columns = make_data_ready(d_f_e, st.session_state.which_model_test)
                df_v = df[t_columns[st.session_state.which_model_test]].dropna()
                df_l =df[f'{st.session_state.which_model_test}_Shifted_Log_Return'].dropna()
                test_msft = df_v
                test_msft_l = df_l
                Ctest = (test_msft_l>0)
                model= serve_model('r_f',st.session_state.which_model_test)
                y_test_pp = model.predict_proba(test_msft)
                accuracy_test = model.score(test_msft, Ctest)
                roc_auc = roc_auc_score(Ctest, y_test_pp[:,1])
                st.write(f'accuracy_test: {accuracy_test}')
                st.write(f'roc_auc: {roc_auc}')
                fpr, tpr, thres = roc_curve(Ctest, y_test_pp[:,1])
                fig, ax = plt.subplots()
                ax = plt.plot(fpr,tpr)
                st.pyplot(fig)
                
        elif st.session_state.trained_model =='Download':
            st.write(st.session_state.trained_model)
            st.text_input('Write one of them (MSFT, AAPL, NVDA, UBER) to see score',key='download_model_test')
            if  st.session_state.download_model_test in ['MSFT', 'AAPL', 'NVDA', 'UBER']:
                model= serve_model('r_f',st.session_state.download_model_test)
                # st.download_button('Click download the model (sklearn)',model)
                output_model = pickle.dumps(model)
                b64 = base64.b64encode(output_model).decode()
                href = f'<a href="data:file/output_model;base64,{b64}" download="myfile.pkl">Download Trained Model .pkl File</a>'
                st.markdown(href, unsafe_allow_html=True)
    elif st.session_state.page_nav =='Logistic_R':
        st.text_input('Train, Test or Download The Model (Train, Test, Download)',key='trained_model')
        if st.session_state.trained_model =='Train':
            st.text_input('Write one of them (MSFT, AAPL, NVDA, UBER) to see score',key='which_model')
            if st.session_state.which_model in ['MSFT', 'AAPL', 'NVDA', 'UBER'] :
                st.write(st.session_state.which_model)
                insert_new_data()
                df,r_f,l_r = train_model_and_save()
                st.markdown('Latest Data')
                st.dataframe(df.head())
                st.write('Train Finished')
                accuracy_train = st.write(f" accuracy_train: { l_r[st.session_state.which_model ][0] }")
                accuracy_test = st.write(f" accuracy_test: { l_r[st.session_state.which_model ][1] }") 
                roc_auc = st.write(f" roc_auc: { l_r[st.session_state.which_model ][2] }")
                fig, ax = plt.subplots()
                ax = plt.plot(l_r[st.session_state.which_model ][3],l_r[st.session_state.which_model ][4])
                st.pyplot(fig)
        elif st.session_state.trained_model =='Test':
            st.file_uploader(label='Try model with the data your own (MSFT, AAPL, NVDA, UBER) ',type='csv',key='test_data')
            d_f_e = pd.read_csv('MSFT.csv')
            st.write('Example input data')
            st.dataframe(d_f_e.head())
            st.text_input('Write one of them (MSFT, AAPL, NVDA, UBER) to see score',key='which_model_test')
            if  st.session_state.which_model_test in ['MSFT', 'AAPL', 'NVDA', 'UBER'] :
                st.write(st.session_state.test_data)
                df,t_columns = make_data_ready(d_f_e, st.session_state.which_model_test)
                df_v = df[t_columns[st.session_state.which_model_test]].dropna()
                df_l =df[f'{st.session_state.which_model_test}_Shifted_Log_Return'].dropna()
                test_msft = df_v
                test_msft_l = df_l
                Ctest = (test_msft_l>0)
                model= serve_model('l_r',st.session_state.which_model_test)
                y_test_pp = model.predict_proba(test_msft)
                accuracy_test = model.score(test_msft, Ctest)
                roc_auc = roc_auc_score(Ctest, y_test_pp[:,1])
                st.write(f'accuracy_test: {accuracy_test}')
                st.write(f'roc_auc: {roc_auc}')
                fpr, tpr, thres = roc_curve(Ctest, y_test_pp[:,1])
                fig, ax = plt.subplots()
                ax = plt.plot(fpr,tpr)
                st.pyplot(fig)
                
        elif st.session_state.trained_model =='Download':
            st.write(st.session_state.trained_model)
            st.text_input('Write one of them (MSFT, AAPL, NVDA, UBER) to see score',key='download_model_test')
            if  st.session_state.download_model_test in ['MSFT', 'AAPL', 'NVDA', 'UBER']:
                model= serve_model('l_r',st.session_state.download_model_test)
                # st.download_button('Click download the model (sklearn)',model)
                output_model = pickle.dumps(model)
                b64 = base64.b64encode(output_model).decode()
                href = f'<a href="data:file/output_model;base64,{b64}" download="myfile.pkl">Download Trained Model .pkl File</a>'
                st.markdown(href, unsafe_allow_html=True)



main()