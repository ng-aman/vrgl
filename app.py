import streamlit as st
from mitosheet.streamlit.v1 import spreadsheet
from pdf_to_investments import get_aggregated_dataframe
import pandas as pd
import base64
import io
from copy import deepcopy

from io import BytesIO

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.close()
    # output.seek(0)
    processed_data = output.getvalue()
    return processed_data


st.set_page_config(layout="wide")

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if 'has_run' not in st.session_state:
        # Getting aggregated data
        df_aggregated_data = get_aggregated_dataframe(uploaded_files)
        st.session_state.data = deepcopy(df_aggregated_data)
        st.session_state.has_run = True
    
    # Display the dataframe in a Mito spreadsheet
    final_dfs, code = spreadsheet(st.session_state.data)
    
    # Option to download final_dfs['df1'] as Excel
    df_xlsx = to_excel(final_dfs['df1'])
    st.download_button(label='Download Excel',
                                    data=df_xlsx ,
                                    file_name= 'results.xlsx')
else:
    st.write("Please upload PDF files to proceed.")
