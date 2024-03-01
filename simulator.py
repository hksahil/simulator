import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy

# Page Settings
st.set_page_config(page_title='Kellogg Dynamic Modelling Simulator',page_icon=':smile:',layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)

# CSS
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Define df1
df1_data = {
    'Project': ['A', 'B'],
    'Region': ['KNA', 'KLA'],
    'NSV': [10, 20],
    'Proj id': [1, 2]
}

df1 = pd.DataFrame(df1_data)
df1_copy=copy.deepcopy(df1)

# Function to add projects
def add_projects(df1, df2):
    # Rename 'nsv' column in df1 to 'df1.nsv'
    df1.rename(columns={'NSV': 'df1.nsv'}, inplace=True)

    # Add a new column df2.nsv which is a replica of df1.nsv
    df1['df2.nsv'] = df1['df1.nsv']

    # Rename 'NSV' column in df2 to 'df2.nsv' and add a new column df1.nsv
    df2.rename(columns={'NSV': 'df2.nsv'}, inplace=True)
    df2['df1.nsv'] = None

    # Concatenate both dataframes
    result = pd.concat([df1, df2], ignore_index=True)

    return result

def plot_bar_graph(df):
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Old NSV', 'New NSV'], y=[df['df1.nsv'].sum(), df['df2.nsv'].sum()], palette=['#003f5c', '#ff6361'])
    plt.xlabel('NSV Type')
    plt.ylabel('NSV ($M)')
    plt.title('NSV of Projects')
    st.pyplot()

def main():
    st.title('Kellogg POC Simulator')

    # Sidebar
    st.sidebar.subheader('Upload simulation Excel file')
    st.sidebar.markdown('Check the input file template [here](https://docs.google.com/spreadsheets/d/1YtrESf2WrrzAlxgdzpo1eyDgjm2X6IUmTZdWz1u406E/edit?usp=sharing).')

    #upload df2
    uploaded_file_df2 = st.sidebar.file_uploader(" ", type=["xlsx", "xls"])

    if uploaded_file_df2:
        df2 = pd.read_excel(uploaded_file_df2)
        df2_copy=copy.deepcopy(df2)

        # Add projects
        result = add_projects(df1, df2)

        # Add filters
        st.sidebar.subheader('Filter Section')
        projects=st.sidebar.multiselect('Project Name',result['Project'].unique(),default=result['Project'].unique())
        regions=st.sidebar.multiselect('Region',result['Region'].unique(),default=result['Region'].unique())
        p_ids=st.sidebar.multiselect('Project ID',result['Proj id'].unique(),default=result['Proj id'].unique())

        # Add a horizontal line
        st.markdown("---")   

        # Filtering your results
        filtered_final_df=result[(result['Project'].isin(projects)) & (result['Region'].isin(regions)) & (result['Proj id'].isin(p_ids))]

        col1,col2=st.columns(2)
        with col1:
            st.write("Original Data")
            st.write(df1_copy)   
        with col2:
            st.write("Uploaded Data")
            st.write(df2_copy) 

        # Plot bar graph
        plot_bar_graph(filtered_final_df)

    else:
        st.warning('Please upload the simulation Excel file to do simulation')

if __name__ == "__main__":
    main()
