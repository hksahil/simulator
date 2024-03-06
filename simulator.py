import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
import duckdb
from streamlit_dynamic_filters import DynamicFilters
from matplotlib import style

# Page Settings
st.set_page_config(page_title='Kellogg Dynamic Modelling Simulator',page_icon=':smile:',layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)

# CSS
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            span.st-emotion-cache-10trblm e1nzilvr1 {color:red !important;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Define df1
df1_data= {
    'Action':['current','current','current'],
    'Project': ['Civic 2.0','Clip ANZ','10 on 10'],
    'Proj_id': [1,2,3],
    'launch_yr': [2026,2018,2019],
    'launch_dt': ['2025-11-10','2018-08-15','2018-12-01'],
    'Region': ['KNA','KAP','KAP'],
    'nsv_yr_1': [5000000,0,1300000],
    'nsv_yr_2': [0,0,0],
    'nsv_yr_3': [0,0,0]
}
df_raw= pd.DataFrame(df1_data)
original_data=copy.deepcopy(df_raw)


def Process_data(df):
# Create both columns using list comprehension
    df['df1_nsv_yr_1'] = [row['df2_nsv_yr_1'] if row['Action'] in ['DELETE', 'CURRENT'] else None for index, row in df.iterrows()]
    df['df1_nsv_yr_2'] = [row['df2_nsv_yr_2'] if row['Action'] in ['DELETE', 'CURRENT'] else None for index, row in df.iterrows()]
    df['df1_nsv_yr_3'] = [row['df2_nsv_yr_3'] if row['Action'] in ['DELETE', 'CURRENT'] else None for index, row in df.iterrows()]
    df['df1_nsv_yr_1_rollup'] = [row['df2_nsv_yr_1_rollup'] if row['Action'] in ['DELETE', 'CURRENT'] else None for index, row in df.iterrows()]
    df['df1_nsv_yr_3_rollup'] = [row['df2_nsv_yr_3_rollup'] if row['Action'] in ['DELETE', 'CURRENT'] else None for index, row in df.iterrows()]
    df['df1_nsv_year'] = [row['df2_nsv_year'] if row['Action'] in ['DELETE', 'CURRENT'] else None for index, row in df.iterrows()]
    df['df1_nsv_wrap'] = [row['df2_nsv_wrap'] if row['Action'] in ['DELETE', 'CURRENT'] else None for index, row in df.iterrows()]
    
    # Deleting the values
    df['df2_nsv_yr_1'] = [None if row['Action'] in ['DELETE'] else row['df2_nsv_yr_1'] for index, row in df.iterrows()]
    df['df2_nsv_yr_2'] = [None if row['Action'] in ['DELETE'] else row['df2_nsv_yr_2'] for index, row in df.iterrows()]
    df['df2_nsv_yr_3'] = [None if row['Action'] in ['DELETE'] else row['df2_nsv_yr_3'] for index, row in df.iterrows()]
    df['df2_nsv_yr_1_rollup'] = [None if row['Action'] in ['DELETE'] else row['df2_nsv_yr_1_rollup'] for index, row in df.iterrows()]
    df['df2_nsv_yr_3_rollup'] = [None if row['Action'] in ['DELETE'] else row['df2_nsv_yr_3_rollup'] for index, row in df.iterrows()]
    df['df2_nsv_year'] = [None if row['Action'] in ['DELETE'] else row['df2_nsv_year'] for index, row in df.iterrows()]
    df['df2_nsv_wrap'] = [None if row['Action'] in ['DELETE'] else row['df2_nsv_wrap'] for index, row in df.iterrows()]

    # Adding 3 year rolling NSV 
    df['df2_nsv_3yr_rollling']= df['df2_nsv_year'].fillna(0)+ df['df2_nsv_wrap'].fillna(0)
    df['df1_nsv_3yr_rollling']= df['df1_nsv_year'].fillna(0)+ df['df1_nsv_wrap'].fillna(0)

    return(df)


def sql_process(df):
    
    # Processing the table 
    z_df1 = duckdb.query("""
            select *  from ( 
            with nsv_calc as (
            select * ,
            row_number() over(partition by Project order by year_of_nsv) filtr ,
            lag(mm_nsv) over (partition by Project order by year_of_nsv ) next_mm_nsv
            from (
            select upper(Action) as Action,Project,Proj_id,launch_dt,Region,nsv_yr_1,nsv_yr_2,nsv_yr_3,
                launch_yr + launch_month_nov_dec_flag launch_year ,
                case
                    when launch_month_nov_dec_flag = 1 then 1
                    else launch_month
                end launch_month ,
                launch_yr + launch_month_nov_dec_flag +
                case
                    when mm_type = 'mm_nsv_yr_1' then 0
                    when mm_type = 'mm_nsv_yr_2' then 1
                    when mm_type = 'mm_nsv_yr_3' then 2
                    when mm_type = 'mm_nsv_yr_dummy' then 3
                    else 0
                end year_of_nsv ,
                mm_nsv *(13-launch_month) yearly_total ,
                mm_nsv *(launch_month-1) mm_nsv ,
                nsv_yr_1_rollup ,
                nsv_yr_3_rollup ,
                curr_yr
            from (             
                select upper(Action) as Action ,Project,Proj_id,launch_dt,Region,nsv_yr_1,nsv_yr_2,nsv_yr_3,
                    extract( year from cast(launch_dt as date)) as launch_yr,
                    extract( month from cast(launch_dt as date)) as launch_month ,
                    case when launch_month in (11, 12) then 1 else 0 end launch_month_nov_dec_flag ,
                    cast(coalesce(nsv_yr_1, 0) as decimal)/ 12 as mm_nsv_yr_1 ,
                    cast(coalesce(nsv_yr_2, 0) as decimal)/ 12 as mm_nsv_yr_2 ,
                    cast(coalesce(nsv_yr_3, 0) as decimal)/ 12 as mm_nsv_yr_3 ,
                    0 as mm_nsv_yr_dummy ,
                    cast(coalesce(nsv_yr_1, 0) as decimal) nsv_yr_1_rollup ,
                    cast(coalesce(nsv_yr_1, 0) as decimal) + cast(coalesce(nsv_yr_2, 0) as decimal) + cast(coalesce(nsv_yr_3, 0) as decimal) nsv_yr_3_rollup,
                    extract(year from current_date) curr_yr
                from
                    df ) unpivot (mm_nsv for mm_type in (mm_nsv_yr_1, mm_nsv_yr_2, mm_nsv_yr_3, mm_nsv_yr_dummy))) )
            select * from nsv_calc )
             """
            ).df()

    #Renaming and dropping few of the columns 
    z_df1.drop(['launch_month','mm_nsv','curr_yr'],axis=1,inplace=True)
    z_df1.rename(columns={'year_of_nsv':'yr_of_nsv', 'launch_year':'launch_yr','filtr':'filter','yearly_total':'df2_nsv_year','next_mm_nsv':'df2_nsv_wrap','nsv_yr_1': 'df2_nsv_yr_1','nsv_yr_2': 'df2_nsv_yr_2','nsv_yr_3': 'df2_nsv_yr_3','nsv_yr_1_rollup': 'df2_nsv_yr_1_rollup','nsv_yr_3_rollup': 'df2_nsv_yr_3_rollup'}, inplace=True)
    

    return(z_df1)

def plot_bar(df):
    yr_nsv = df.groupby('Region')[['df1_nsv_yr_3_rollup', 'df2_nsv_yr_3_rollup']].sum().reset_index()
    #st.write(yr_nsv)
    #st.write('graph table',yr_nsv_sales)
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    # Width of the bars
    bar_width = 0.2
    # Index for the x-axis
    ind = range(len(yr_nsv))
    # Plotting old sales
    old_sales = ax.bar(ind, yr_nsv['df1_nsv_yr_3_rollup']/4, bar_width, label='Old Sales')
    new_sales = ax.bar([i + bar_width for i in ind], yr_nsv['df2_nsv_yr_3_rollup']/4, bar_width, label='New Sales')
    # Setting labels and title
    ax.set_xlabel('Year of NSV')
    ax.set_ylabel('NSV 3 year rolling')
    ax.set_title('Old vs New NSV by Year of NSV')
    ax.set_xticks([i + bar_width / 2 for i in ind])
    ax.set_xticklabels(yr_nsv['Region'])
    #ax.set(ylim=(10, 150))
    ax.legend()
    # Show plot
    st.pyplot()

def plot_bar2(df):
    yr_nsv = df.groupby('yr_of_nsv')[['df1_nsv_yr_3_rollup', 'df2_nsv_yr_3_rollup']].sum().reset_index()
    #st.write(yr_nsv)
    #st.write('graph table',yr_nsv_sales)
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    # Width of the bars
    bar_width = 0.2
    # Index for the x-axis
    ind = range(len(yr_nsv))
    # Plotting old sales
    old_sales = ax.bar(ind, yr_nsv['df1_nsv_yr_3_rollup']/4, bar_width, label='Old Sales')
    new_sales = ax.bar([i + bar_width for i in ind], yr_nsv['df2_nsv_yr_3_rollup']/4, bar_width, label='New Sales')
    # Setting labels and title
    ax.set_xlabel('Year of NSV')
    ax.set_ylabel('NSV 3 year rolling')
    ax.set_title('Old vs New NSV by Year of NSV')
    ax.set_xticks([i + bar_width / 2 for i in ind])
    ax.set_xticklabels(yr_nsv['yr_of_nsv'])
    #ax.set(ylim=(0,60))
    ax.legend()
    # Show plot
    st.pyplot()
     
def main():
    #st.title('Kellogg POC Simulator')
    st.markdown("<span style='color:#f60b45;font-size:44px;font-family:Source Sans Pro;font-weight:700'>Pre Data Dynamic Modelling Simulator</span>",
             unsafe_allow_html=True)

    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/Kellanova_logo.svg/1200px-Kellanova_logo.svg.png",width=150)
    
    expander = st.sidebar.expander("How to use this Tool?")
    expander.write("Download the file containing pre data from the button below")
    expander.write("User can edit this file to add/delete Projects.")
    expander.write("To Add Projects, mention ADD in Action column.")
    expander.write("To Delete Projects, change the Action value to DELETE ")
    #st.sidebar.markdown("----")
    #st.sidebar.markdown('Download the excel to perform the Actions. [Download](https://docs.google.com/spreadsheets/d/1F6Dd7_MgyqBCnsy1UbQPLk5NRw_TSaEDqazrCAojtMg/edit?usp=sharing)')
    # Download Functionality
    st.sidebar.download_button("Download the Pre data",df_raw.to_csv(index=False),file_name="Pre_Launch Data.csv",mime="text/csv",help='mmmm')
    st.sidebar.subheader('Upload the new Pre data file')

    #upload user input file
    uploaded_file_df1 = st.sidebar.file_uploader(" ", type=["xlsx", "xls","csv"])

    if uploaded_file_df1:
            df1 = pd.read_csv(uploaded_file_df1)

            # changing the data type and format of launch date
            df1['launch_dt'] = pd.to_datetime(df1['launch_dt'])
            df1['launch_dt'] = df1['launch_dt'].dt.strftime('%Y-%m-%d')
            user_input_data = copy.deepcopy(df1)

            result = sql_process(df1)
            st.write('Processed Data',Process_data(result))

            col1,col2=st.columns(2) 
            with col1:
                plot_bar(result)
            with col2:
                plot_bar2(result)

    
            # st.write(df1)
    

if __name__ == "__main__":
    main()
