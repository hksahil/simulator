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

if "load_state" not in st.session_state:
    st.session_state.load_state=False
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
def old_calculate_rolling_nsv(row):
    if row['launch_yr'] == row['yr_of_nsv']:
        return row['old_nsv_year']
    elif row['launch_yr'] == row['yr_of_nsv'] - 1:
        return row['old_nsv_wrap']
    else:
        return 0
    
def new_calculate_rolling_nsv(row):
    if row['launch_yr'] == row['yr_of_nsv']:
        return row['new_nsv_year']
    elif row['launch_yr'] == row['yr_of_nsv'] - 1:
        return row['new_nsv_wrap']
    else:
        return 0
    
def remove_decimal(number):
    if number.endswith('.0'):
        return number[:-2]
    else:
        return number

def Process_data(df,dfx): 

    dfx.rename(columns={'fltr':'filter'}, inplace=True )
    # Deleting the values
    df['new_nsv_yr_1'] = [None if row['flag'] == False  else row['new_nsv_yr_1'] for index, row in df.iterrows()]
    df['new_nsv_yr_2'] = [None if row['flag'] == False else row['new_nsv_yr_2'] for index, row in df.iterrows()]
    df['new_nsv_yr_3'] = [None if row['flag'] == False  else row['new_nsv_yr_3'] for index, row in df.iterrows()]
    df['new_nsv_yr_1_rollup'] = [None if row['flag'] ==False else row['new_nsv_yr_1_rollup'] for index, row in df.iterrows()]
    df['new_nsv_yr_3_rollup'] = [None if row['flag'] ==False  else row['new_nsv_yr_3_rollup'] for index, row in df.iterrows()]
    df['new_nsv_year'] = [None if row['flag'] ==False else row['new_nsv_year'] for index, row in df.iterrows()]
    df['new_nsv_wrap'] = [None if row['flag'] ==False else row['new_nsv_wrap'] for index, row in df.iterrows()]

    dfx['new_yr_of_nsv'] = [remove_decimal(str(num)) for num in dfx['yr_of_nsv']]
    df['new_yr_of_nsv'] = [remove_decimal(str(num)) for num in df['yr_of_nsv']]

    #join measures from original to input file 
    df['Combined'] = df['new_yr_of_nsv'] + "_" + df['Project']
    dfx['Combined'] = dfx['new_yr_of_nsv'] + "_" + dfx['Project']
    
    # joining
    result_concat = df.join(dfx.set_index("Combined"),how = 'left' ,rsuffix="_df1", on="Combined")

    result_concat.drop(['Project_df1','Region_df1','Proj_id_df1','yr_of_nsv_df1','launch_dt_df1','launch_yr_df1','new_yr_of_nsv','new_yr_of_nsv_df1'],axis=1, inplace=True )
    result_concat.rename(columns={'nsv_yr':'old_nsv_year','nsv_wrap':'old_nsv_wrap','nsv_yr_1': 'old_nsv_yr_1','nsv_yr_2': 'old_nsv_yr_2','nsv_yr_3': 'old_nsv_yr_3','nsv_yr_1_rollup': 'old_nsv_yr_1_rollup','nsv_yr_3_rollup': 'old_nsv_yr_3_rollup'}, inplace=True )

    # Adding 3 year rolling NSV 
    result_concat['new_nsv_3yr_rollling']= result_concat['new_nsv_year'].fillna(0)+ result_concat['new_nsv_wrap'].fillna(0)
    result_concat['old_nsv_3yr_rollling']= result_concat['old_nsv_year'].fillna(0)+ result_concat['old_nsv_wrap'].fillna(0)
    result_concat['new_nsv_3yr_rollling'] = [None if row['flag'] ==False else row['new_nsv_3yr_rollling'] for index, row in result_concat.iterrows()]

    # Apply the function to create the column
    result_concat['old_1 Year Rolling NSV'] = result_concat.apply(old_calculate_rolling_nsv, axis=1)
    result_concat['new_1 Year Rolling NSV'] = result_concat.apply(new_calculate_rolling_nsv, axis=1)
    result_concat['new_1 Year Rolling NSV'] = [None if row['flag'] ==False else row['new_1 Year Rolling NSV'] for index, row in result_concat.iterrows()]

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
    z_df1.rename(columns={'year_of_nsv':'yr_of_nsv', 'launch_year':'launch_yr','filtr':'filter','yearly_total':'new_nsv_year','next_mm_nsv':'new_nsv_wrap','nsv_yr_1': 'new_nsv_yr_1','nsv_yr_2': 'new_nsv_yr_2','nsv_yr_3': 'new_nsv_yr_3','nsv_yr_1_rollup': 'new_nsv_yr_1_rollup','nsv_yr_3_rollup': 'new_nsv_yr_3_rollup'}, inplace=True)
    return(z_df1)

def plot_bar(df,measure):
    if measure == "1 year rolling nsv":
        old_column = 'old_1 Year Rolling NSV'
        new_column = 'new_1 Year Rolling NSV'
        ylabel = '1 year nsv'
        title = 'Old vs New NSV by Region (1 Year Rolling)'
    elif measure == "3 year rolling nsv":
        old_column = 'old_nsv_3yr_rollling'
        new_column = 'new_nsv_3yr_rollling'
        ylabel = '3 year nsv'
        title = 'Old vs New NSV by Region (3 Year Rolling)'

    df[old_column]=df[old_column].round(2)
    df[new_column]=df[new_column].round(2)
    yr_nsv = df.groupby('Region')[[old_column,new_column]].sum().reset_index()
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    # Width of the bars
    bar_width = 0.2
    # Index for the x-axis
    ind = range(len(yr_nsv))
    # Plotting old sales
    old_sales = ax.bar(ind, yr_nsv[old_column]/4, bar_width, label='Old Sales')
    new_sales = ax.bar([i + bar_width for i in ind], yr_nsv[new_column]/4, bar_width, label='New Sales')
    # Setting labels and title
    ax.set_xlabel('Region')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([i + bar_width / 2 for i in ind])
    ax.set_xticklabels(yr_nsv['Region'])
    #ax.set(ylim=(10, 150))
    ax.legend()
    # Show plot
    st.pyplot()
    yr_nsv['Difference'] = yr_nsv[new_column] - yr_nsv[old_column]
    st.dataframe(yr_nsv,height=175)

def plot_bar2(df, measure):
    if measure == "1 year rolling nsv":
        old_column = 'old_1 Year Rolling NSV'
        new_column = 'new_1 Year Rolling NSV'
        ylabel = '1 year nsv'
        title = 'Old vs New NSV by Region (1 Year Rolling)'
    elif measure == "3 year rolling nsv":
        old_column = 'old_nsv_3yr_rollling'
        new_column = 'new_nsv_3yr_rollling'
        ylabel = '3 year nsv'
        title = 'Old vs New NSV by Region (3 Year Rolling)'

    df[old_column]=df[old_column].round(2)
    df[new_column]=df[new_column].round(2)
    yr_nsv = df.groupby('yr_of_nsv')[[old_column,new_column]].sum().reset_index()
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    # Width of the bars
    bar_width = 0.2
    # Index for the x-axis
    ind = range(len(yr_nsv))
    # Plotting old sales
    old_sales = ax.bar(ind, yr_nsv[old_column]/4, bar_width, label='Old Sales')
    new_sales = ax.bar([i + bar_width for i in ind], yr_nsv[new_column]/4, bar_width, label='New Sales')
    # Setting labels and title
    ax.set_xlabel('yr_of_nsv')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([i + bar_width / 2 for i in ind])
    ax.set_xticklabels(yr_nsv['yr_of_nsv'])
    #ax.set(ylim=(10, 150))
    ax.legend()
    # Show plot
    st.pyplot()
    yr_nsv['Difference'] = yr_nsv[new_column] - yr_nsv[old_column]
    st.dataframe(yr_nsv,height=175)

def validate(df):
    # duplicate project names 
    errors = []
    #Validate column names
    expected_columns = ['Action', 'Proj_id', 'Project', 'launch_yr', 'launch_dt', 'Region', 'nsv_yr_1', 'nsv_yr_2', 'nsv_yr_3']
    if list(df.columns) != expected_columns:
        errors.append("Error: Column names should not change. \n\n \t\t Please keep the column name as same as the template")
        return errors

    # Validate no duplicate project names
    if df['Project'].duplicated().any():
       errors.append("Error: The Projects you entered already exist. \n\n \t\t Please change the new project name to run the simulator", )

   # Validate no nulls in project names, project ids
    columns_to_check = ['Project', 'Proj_id']
    for column in columns_to_check:
        if df[column].isnull().any():
            errors.append(f"Error: There are null values in column '{column}'. \n\n  \t\t Please provide the values.")
        
    #Validate for string values in measure
    if df['nsv_yr_1'].dtype == 'object' or df['nsv_yr_2'].dtype == 'object' or df['nsv_yr_3'].dtype == 'object' :
        errors.append(f"Error: There are non-numerical values in NSV columns.\n\n  \t\t Please provide the numerical values.")
    
    return errors

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
        df1.sort_values(by='Action',ascending=True,inplace=True)
        if len(validate(df1))>0:
            for i in validate(df1):
                st.error(i)
        else :
            # changing the data type and format of launch date
            df1['launch_dt'] = pd.to_datetime(df1['launch_dt'])
            df1['launch_dt'] = df1['launch_dt'].dt.strftime('%Y-%m-%d')
            user_input_data = copy.deepcopy(df1)
            dynamic_filters = DynamicFilters(df1, filters=['Project', 'Region'])
            dynamic_filters.display_filters(location='sidebar')
            df_filtered = dynamic_filters.filter_df()

            # Checkbox 
            df_filtered['flag']=True
            df_filtered.insert(0, 'flag', df_filtered.pop('flag'))
            df_filtered.insert(1, 'Project', df_filtered.pop('Project'))
            q=st.data_editor(df_filtered,    column_config={
            "flag": st.column_config.CheckboxColumn(
                "Include",
                help="Select the projects you want to delete",
                default=False,
            )},disabled=['Project','Proj_id','Action','launch_yr','launch_dt','Region','nsv_yr_1','nsv_yr_2','nsv_yr_3'],hide_index=True,)
            #df_deletion_new=q[q['flag']==False]
            #st.write('q',q)

            # Button to simulate
            button_pressed = st.button("Start the Simulation")

            if button_pressed or st.session_state.load_state:
                st.session_state.load_state=True
                # Deleting the user selected rows 
                sql_result = sql_process(q)
                final = Process_data(sql_result,raw_copy)
                final.sort_values(by='Proj_id',ascending=False,inplace=True)
                final.drop(['flag'],axis=1,inplace=True)

                #Dropdown for the measure
                measure_selected = st.selectbox("Select the measure: ",('1 year rolling nsv', '3 year rolling nsv'))
                #st.write(measure_selected)

                col1,col2=st.columns(2) 
                with col1:
                    plot_bar(final,measure_selected)
                with col2:
                    plot_bar2(final,measure_selected)
                st.write('The Processed Data',final)
    

if __name__ == "__main__":
    main()
