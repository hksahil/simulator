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
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Define df1
df_raw = pd.read_csv('pre.csv')
df_raw.rename(columns = {'proj_nm':'Project','proj_id':'Proj_id','geog_short':'Region'},inplace=True)
raw_copy=copy.deepcopy(df_raw) 
df_raw = df_raw[df_raw['fltr']==1]
df_raw = df_raw[['Proj_id','Project','launch_yr','launch_dt','Region','nsv_yr_1','nsv_yr_2','nsv_yr_3']]
df_raw['Action']='Current'
df_raw.set_index('Action', inplace=True)
df_raw.sort_values(by='Proj_id',ascending=True,inplace=True)

# To calculate 1 Year Rolling NSV
def df1_calculate_rolling_nsv(row):
    if row['launch_yr'] == row['yr_of_nsv']:
        #print(result_concat['df1_launch_yr'],result_concat['df1_yr_of_nsv'],'in first loop',)
        return row['df1_nsv_year']
    elif row['launch_yr'] == row['yr_of_nsv'] - 1:
        #print(result_concat['df1_launch_yr'],result_concat['df1_yr_of_nsv'],'in second loop',)
        return row['df1_nsv_wrap']
    else:
        #print(result_concat['df1_launch_yr'],result_concat['df1_yr_of_nsv'],'in third loop',)
        return 0
    
def df2_calculate_rolling_nsv(row):
    if row['launch_yr'] == row['yr_of_nsv']:
        #print(result_concat['df1_launch_yr'],result_concat['df1_yr_of_nsv'],'in first loop',)
        return row['df2_nsv_year']
    elif row['launch_yr'] == row['yr_of_nsv'] - 1:
        #print(result_concat['df1_launch_yr'],result_concat['df1_yr_of_nsv'],'in second loop',)
        return row['df2_nsv_wrap']
    else:
        #print(result_concat['df1_launch_yr'],result_concat['df1_yr_of_nsv'],'in third loop',)
        return 0
    
def Process_data(df,dfx): 

    dfx.rename(columns={'fltr':'filter'}, inplace=True )
    # Deleting the values
    df['df2_nsv_yr_1'] = [None if row['flag'] == True  else row['df2_nsv_yr_1'] for index, row in df.iterrows()]
    df['df2_nsv_yr_2'] = [None if row['flag'] == True else row['df2_nsv_yr_2'] for index, row in df.iterrows()]
    df['df2_nsv_yr_3'] = [None if row['flag'] == True  else row['df2_nsv_yr_3'] for index, row in df.iterrows()]
    df['df2_nsv_yr_1_rollup'] = [None if row['flag'] ==True else row['df2_nsv_yr_1_rollup'] for index, row in df.iterrows()]
    df['df2_nsv_yr_3_rollup'] = [None if row['flag'] ==True  else row['df2_nsv_yr_3_rollup'] for index, row in df.iterrows()]
    df['df2_nsv_year'] = [None if row['flag'] ==True else row['df2_nsv_year'] for index, row in df.iterrows()]
    df['df2_nsv_wrap'] = [None if row['flag'] ==True else row['df2_nsv_wrap'] for index, row in df.iterrows()]

    #join measures from original to input file 
    df['Combined'] = df['yr_of_nsv'].astype(str).str.rstrip('.0') + "_" + df['Project']
    dfx['Combined'] = dfx['yr_of_nsv'].astype(str).str.rstrip('.0') + "_" + dfx['Project']

    result_concat = df.join(dfx.set_index("Combined"),how = 'left' ,rsuffix="_df1", on="Combined" )
    result_concat.drop(['Project_df1','Region_df1','Proj_id_df1','yr_of_nsv_df1','launch_dt_df1','launch_yr_df1'],axis=1, inplace=True )
    result_concat.rename(columns={'nsv_yr':'df1_nsv_year','nsv_wrap':'df1_nsv_wrap','nsv_yr_1': 'df1_nsv_yr_1','nsv_yr_2': 'df1_nsv_yr_2','nsv_yr_3': 'df1_nsv_yr_3','nsv_yr_1_rollup': 'df1_nsv_yr_1_rollup','nsv_yr_3_rollup': 'df1_nsv_yr_3_rollup'}, inplace=True )

    # Adding 3 year rolling NSV 
    result_concat['df2_nsv_3yr_rollling']= result_concat['df2_nsv_year'].fillna(0)+ result_concat['df2_nsv_wrap'].fillna(0)
    result_concat['df1_nsv_3yr_rollling']= result_concat['df1_nsv_year'].fillna(0)+ result_concat['df1_nsv_wrap'].fillna(0)
    result_concat['df2_nsv_3yr_rollling'] = [None if row['flag'] ==True else row['df2_nsv_3yr_rollling'] for index, row in result_concat.iterrows()]

    # Apply the function to create the column
    result_concat['df1_1 Year Rolling NSV'] = result_concat.apply(df1_calculate_rolling_nsv, axis=1)
    result_concat['df2_1 Year Rolling NSV'] = result_concat.apply(df2_calculate_rolling_nsv, axis=1)
    result_concat['df2_1 Year Rolling NSV'] = [None if row['flag'] ==True else row['df2_1 Year Rolling NSV'] for index, row in result_concat.iterrows()]

    return(result_concat)


def sql_process(df):
    
    # Processing the table 
    z_df1 = duckdb.query("""
            select *  from ( 
            with nsv_calc as (
            select * ,
            row_number() over(partition by Project order by year_of_nsv) filtr ,
            lag(mm_nsv) over (partition by Project order by year_of_nsv ) next_mm_nsv
            from (
            select flag,upper(Action) as Action,Project,Proj_id,launch_dt,Region,nsv_yr_1,nsv_yr_2,nsv_yr_3,
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
                select flag,upper(Action) as Action ,Project,Proj_id,launch_dt,Region,nsv_yr_1,nsv_yr_2,nsv_yr_3,
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
    st.write(df['Project'].count())
    yr_nsv = df.groupby('Region')[['df1_nsv_3yr_rollling', 'df2_nsv_3yr_rollling']].sum().reset_index()
    st.write(yr_nsv)
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    # Width of the bars
    bar_width = 0.2
    # Index for the x-axis
    ind = range(len(yr_nsv))
    # Plotting old sales
    old_sales = ax.bar(ind, yr_nsv['df1_nsv_3yr_rollling']/4, bar_width, label='Old Sales')
    new_sales = ax.bar([i + bar_width for i in ind], yr_nsv['df2_nsv_3yr_rollling']/4, bar_width, label='New Sales')
    # Setting labels and title
    ax.set_xlabel('Region')
    ax.set_ylabel('NSV 3 year rolling')
    ax.set_title('Old vs New NSV by Region')
    ax.set_xticks([i + bar_width / 2 for i in ind])
    ax.set_xticklabels(yr_nsv['Region'])
    #ax.set(ylim=(10, 150))
    ax.legend()
    # Show plot
    st.pyplot()

def plot_bar2(df):
    yr_nsv = df.groupby('yr_of_nsv')[['df1_nsv_3yr_rollling', 'df2_nsv_3yr_rollling']].sum().reset_index()
    st.write(yr_nsv)
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    # Width of the bars
    bar_width = 0.2
    # Index for the x-axis
    ind = range(len(yr_nsv))
    # Plotting old sales
    old_sales = ax.bar(ind, yr_nsv['df1_nsv_3yr_rollling']/4, bar_width, label='Old Sales')
    new_sales = ax.bar([i + bar_width for i in ind], yr_nsv['df2_nsv_3yr_rollling']/4, bar_width, label='New Sales')
    # Setting labels and title
    ax.set_xlabel('Original Year of NSV')
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
    # Download Functionality
    st.sidebar.download_button("Download the Pre Launch Data",df_raw.to_csv(index=True),file_name="Pre_Launch Data.csv",mime="text/csv")
    st.sidebar.subheader('Upload simulation Excel file')

    #upload user input file
    uploaded_file_df1 = st.sidebar.file_uploader(" ", type=["csv"])

    if uploaded_file_df1:
            df1 = pd.read_csv(uploaded_file_df1)

            # changing the data type and format of launch date
            df1['launch_dt'] = pd.to_datetime(df1['launch_dt'])
            df1['launch_dt'] = df1['launch_dt'].dt.strftime('%Y-%m-%d')
            user_input_data = copy.deepcopy(df1)
            dynamic_filters = DynamicFilters(df1, filters=['Project', 'Region'])
            dynamic_filters.display_filters(location='sidebar')
            df_filtered = dynamic_filters.filter_df()

            # Checkbox 
            df_filtered['flag']=False
            df_filtered.insert(0, 'flag', df_filtered.pop('flag'))
            df_filtered.insert(1, 'Project', df_filtered.pop('Project'))
            q=st.data_editor(df_filtered,    column_config={
            "flag": st.column_config.CheckboxColumn(
                "Delete",
                help="Select the projects you want to delete",
                default=True,
            )},disabled=['Project','Proj_id','Action','launch_yr','launch_dt','Region','nsv_yr_1','nsv_yr_2','nsv_yr_3'],hide_index=True,)
            #df_deletion_new=q[q['flag']==False]
            #st.write('q',q)

            # Button to simulate
            button_pressed = st.button("Delete the Selection")

            if button_pressed:
                # Deleting the user selected rows 
                sql_result = sql_process(q)
                final = Process_data(sql_result,raw_copy)
                st.write('The Processed Data',final)

                col1,col2=st.columns(2) 
                with col1:
                    plot_bar(final)
                with col2:
                    plot_bar2(final)
    

if __name__ == "__main__":
    main()
