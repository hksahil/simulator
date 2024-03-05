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
df1_data = {
    'launch_yr': [2026,2026,2026,2026,2018,2018,2018,2018,2019,2019,2019,2019],
    'launch_dt': ['2025-11-10','2025-11-10','2025-11-10','2025-11-10','2018-08-15','2018-08-15','2018-08-15','2018-08-15','2018-12-01','2018-12-01','2018-12-01','2018-12-01'],
    'Project': ['Civic 2.0','Civic 2.0','Civic 2.0','Civic 2.0', 'Clip ANZ', 'Clip ANZ', 'Clip ANZ', 'Clip ANZ','10 on 10','10 on 10','10 on 10','10 on 10'],
    'Region': ['KNA','KNA','KNA','KNA', 'KAP','KAP','KAP','KAP','KAP','KAP','KAP','KAP'],
    'nsv_yr_1': [5000000,5000000,5000000,5000000,0,0,0,0,1300000,1300000,1300000,1300000],
    'nsv_yr_2': [0,0,0,0,0,0,0,0,0,0,0,0],
    'nsv_yr_3': [0,0,0,0,0,0,0,0,0,0,0,0],
    'Proj_id': [1,1,1,1,2,2,2,2,3,3,3,3],
    'nsv_yr_1_rollup':[5000000,5000000,5000000,5000000,0,0,0,0,1300000,1300000,1300000,1300000],
    'nsv_yr_3_rollup':[5000000,5000000,5000000,5000000,0,0,0,0,1300000,1300000,1300000,1300000],
    'filter':[1,2,3,4,1,2,3,4,1,2,3,4],
    'nsv_year':[833333.3332,0,0,0,0,0,0,0,108333.3333,0,0,0],
    'nsv_wrap':[0,4166666.666,0,0,0,0,0,0,0,1191666.6663,0,0],
    'yr_of_nsv':[2026,2027,2028,2029,2018,2019,2020,2021,2019,2020,2021,2022]
}

df1 = pd.DataFrame(df1_data)
df1_copy=copy.deepcopy(df1)



# Function to add projects
def add_projects(df1, df2):
    # Rename columns in df1 to 'df1.column_name'
    df1.rename(columns={'nsv_yr_1': 'df1_nsv_yr_1'}, inplace=True)
    df1.rename(columns={'nsv_yr_2': 'df1_nsv_yr_2'}, inplace=True)
    df1.rename(columns={'nsv_yr_3': 'df1_nsv_yr_3'}, inplace=True)
    df1.rename(columns={'nsv_yr_1_rollup': 'df1_nsv_yr_1_rollup'}, inplace=True)
    df1.rename(columns={'nsv_yr_3_rollup': 'df1_nsv_yr_3_rollup'}, inplace=True)
    df1.rename(columns={'nsv_year': 'df1_nsv_year'}, inplace=True)
    df1.rename(columns={'nsv_wrap': 'df1_nsv_wrap'}, inplace=True)


    # Add a new column df2.nsv which is a replica of df1.nsv
    df1['df2_nsv_yr_1'] = df1['df1_nsv_yr_1']
    df1['df2_nsv_yr_2'] = df1['df1_nsv_yr_2']
    df1['df2_nsv_yr_3'] = df1['df1_nsv_yr_3']
    df1['df2_nsv_yr_1_rollup'] = df1['df1_nsv_yr_1_rollup']
    df1['df2_nsv_yr_3_rollup'] = df1['df1_nsv_yr_3_rollup']
    df1['df2_nsv_year'] = df1['df1_nsv_year']
    df1['df2_nsv_wrap'] = df1['df1_nsv_wrap']

    # Rename columns in df2 to 'df2.columns'
    df2.rename(columns={'nsv_yr_1': 'df2_nsv_yr_1'}, inplace=True)
    df2.rename(columns={'nsv_yr_2': 'df2_nsv_yr_2'}, inplace=True)
    df2.rename(columns={'nsv_yr_3': 'df2_nsv_yr_3'}, inplace=True)

    #add a new columns df1.columns
    df2['df1_nsv_yr_1'] = None
    df2['df1_nsv_yr_2'] = None
    df2['df1_nsv_yr_3'] = None
    df2['df1_nsv_yr_1_rollup'] = None
    df2['df1_nsv_yr_3_rollup'] = None
    df2['df1_nsv_year'] = None
    df2['df1_nsv_wrap'] = None

    # Concatenate both dataframes
    result = pd.concat([df1, df2], ignore_index=True)

    return result

def plot_bar_graph(df):

    yr_nsv_sales = df.groupby('yr_of_nsv')[['df1_3_yr_nsv_for_yr_of_nsv', 'df2_3_yr_nsv_for_yr_of_nsv']].sum().reset_index()
    #st.write('graph table',yr_nsv_sales)
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    # Width of the bars
    bar_width = 0.2
    # Index for the x-axis
    ind = range(len(yr_nsv_sales))
    # Plotting old sales
    old_sales = ax.bar(ind, yr_nsv_sales['df1_3_yr_nsv_for_yr_of_nsv'], bar_width, label='Old Sales')
    new_sales = ax.bar([i + bar_width for i in ind], yr_nsv_sales['df2_3_yr_nsv_for_yr_of_nsv'], bar_width, label='New Sales')
    # Setting labels and title
    ax.set_xlabel('Year of NSV')
    ax.set_ylabel('NSV 3 year rolling')
    ax.set_title('Old vs New NSV by Year of NSV')
    ax.set_xticks([i + bar_width / 2 for i in ind])
    ax.set_xticklabels(yr_nsv_sales['yr_of_nsv'])
    ax.legend()
    # Show plot
    st.pyplot()

def plot_bar_region(df):
    Region_grp = df.groupby('Region')[['df1_3_yr_nsv_for_yr_of_nsv', 'df2_3_yr_nsv_for_yr_of_nsv']].sum().reset_index()
    #st.write('graph table',Region_grp)
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    # Width of the bars
    bar_width = 0.2
    # Index for the x-axis
    ind = range(len(Region_grp))
    # Plotting old sales
    old_sales = ax.bar(ind, Region_grp['df1_3_yr_nsv_for_yr_of_nsv'], bar_width, label='Old Sales')
    new_sales = ax.bar([i + bar_width for i in ind], Region_grp['df2_3_yr_nsv_for_yr_of_nsv'], bar_width, label='New Sales')
    # Setting labels and title
    ax.set_xlabel('Region')
    ax.set_ylabel('NSV 3 year rolling')
    ax.set_title('Old vs New NSV by Region')
    ax.set_xticks([i + bar_width / 2 for i in ind])
    ax.set_xticklabels(Region_grp['Region'])
    ax.legend()
    # Show plot
    st.pyplot()    


def main():
    st.title('Kellogg POC Simulator')

    # Sidebar
    st.sidebar.subheader('Upload simulation Excel file')
    st.sidebar.markdown('Check the input file template [here](https://docs.google.com/spreadsheets/d/1YtrESf2WrrzAlxgdzpo1eyDgjm2X6IUmTZdWz1u406E/edit?usp=sharing).')

    #upload user input file
    uploaded_file_df2 = st.sidebar.file_uploader(" ", type=["xlsx", "xls"])

    if uploaded_file_df2:
        df2 = pd.read_excel(uploaded_file_df2)
        # changing the data type and format of launch date
        df2['launch_dt'] = pd.to_datetime(df2['launch_dt'])
        df2['launch_dt'] = df2['launch_dt'].dt.strftime('%Y-%m-%d')
        df2_copy=copy.deepcopy(df2)

        #st.write(df1)
        #st.write(df2)

        z = duckdb.query("""
        select *  from ( 
        with nsv_calc as (
        select * ,
		row_number() over(partition by Project order by year_of_nsv) filtr ,
		lag(mm_nsv) over (partition by Project order by year_of_nsv ) next_mm_nsv
        from (
		select Project,Proj_id,launch_dt,Region,nsv_yr_1,nsv_yr_2,nsv_yr_3,
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
			df2_nsv_yr_1_rollup ,
			df2_nsv_yr_3_rollup ,
			curr_yr
		from (             
			select Project,Proj_id,launch_dt,Region,nsv_yr_1,nsv_yr_2,nsv_yr_3,
                extract( year from cast(launch_dt as date)) as launch_yr,
                extract( month from cast(launch_dt as date)) as launch_month ,
				case when launch_month in (11, 12) then 1 else 0 end launch_month_nov_dec_flag ,
                cast(coalesce(nsv_yr_1, 0) as decimal)/ 12 as mm_nsv_yr_1 ,
                cast(coalesce(nsv_yr_1, 0) as decimal)/ 12 as mm_nsv_yr_1 ,
				cast(coalesce(nsv_yr_2, 0) as decimal)/ 12 as mm_nsv_yr_2 ,
				cast(coalesce(nsv_yr_3, 0) as decimal)/ 12 as mm_nsv_yr_3 ,
				0 as mm_nsv_yr_dummy ,
				cast(coalesce(nsv_yr_1, 0) as decimal) df2_nsv_yr_1_rollup ,
				cast(coalesce(nsv_yr_1, 0) as decimal) + cast(coalesce(nsv_yr_2, 0) as decimal) + cast(coalesce(nsv_yr_3, 0) as decimal) df2_nsv_yr_3_rollup,
			    extract(year from current_date) curr_yr
			from
				df2 ) unpivot (mm_nsv for mm_type in (mm_nsv_yr_1, mm_nsv_yr_2, mm_nsv_yr_3, mm_nsv_yr_dummy))) )
	select * from nsv_calc )
    """
        ).df()

        #Renaming and dropping few of the columns 
        z.drop(['launch_month','mm_nsv','curr_yr'],axis=1,inplace=True)
        z.rename(columns={'year_of_nsv':'yr_of_nsv', 'launch_year':'launch_yr','filtr':'filter','yearly_total':'df2_nsv_year','next_mm_nsv':'df2_nsv_wrap'}, inplace=True)
        #st.write(z)
        
        # Creating a column just for Year of NSV to show its NSV Value
        result = add_projects(df1, z)
        result['df2_3_yr_nsv_for_yr_of_nsv']= result['df2_nsv_year']+result['df2_nsv_wrap']
        result['df1_3_yr_nsv_for_yr_of_nsv']= result['df1_nsv_year']+result['df1_nsv_wrap']

        dynamic_filters = DynamicFilters(result, filters=['Project', 'Region', 'Proj_id'])
        dynamic_filters.display_filters(location='sidebar')
        df_filtered = dynamic_filters.filter_df()
        st.write('The Processed Data',df_filtered)
        st.markdown("----")

        # #variable for NSV Rollings
        # df2_nsv_1yr_rolling = result.loc[result['filter'] == 1, 'df2_nsv_year'].sum() + result.loc[result['filter'] == 2, 'df2_nsv_wrap'].sum()
        # df2_nsv_3yr_rolling = result['df2_nsv_year'].sum() +  result['df2_nsv_wrap'].sum()
        # df1_nsv_1yr_rolling = result.loc[result['filter'] == 1, 'df1_nsv_year'].sum() + result.loc[result['filter'] == 2, 'df1_nsv_wrap'].sum()
        # df1_nsv_3yr_rolling = result['df1_nsv_year'].sum() +  result['df1_nsv_wrap'].sum()     

        # Plot bar graph
        col1,col2=st.columns(2) 
        with col1:
            plot_bar_graph(df_filtered) 
        with col2:
            plot_bar_region(df_filtered)

    else:
        st.warning('Please upload the simulation Excel file to do simulation')

if __name__ == "__main__":
    main()
