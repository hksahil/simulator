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
df_raw = pd.read_csv('pre_1.csv')
df_raw = df_raw.dropna(subset=['launch_dt'])
df_raw = df_raw[df_raw['stage_gate']!='Hold']
df_raw.rename(columns = {'proj_nm':'Project','proj_id':'Proj_id','geog_short':'Region','bu_short':'bu',
                         'prtflo_bkt':'porfolio_bucket','inc_nsv_yr_1':'insv_yr_1','inc_nsv_yr_2':'insv_yr_2',
                         'inc_nsv_yr_3':'insv_yr_3'},inplace=True)
raw_copy=copy.deepcopy(df_raw)
df_raw = df_raw[['fltr','Proj_id','launch_yr','rd_days','Region','bu','Project','proj_desc','porfolio_bucket',
                 'pd_or_ft_days','pkg_days','canblz_pct','gross_mrgn_pct',
                 'nsv_yr_1','nsv_yr_2','nsv_yr_3','insv_yr_1','insv_yr_2','insv_yr_3','launch_dt',
                 'gsv_yr_1','gsv_yr_2','gsv_yr_3']]
df_raw = df_raw[df_raw['fltr']==1]
df_raw = df_raw.drop(['fltr'], axis=1)
df_raw['Action']='Current'
df_raw_copy = copy.deepcopy(df_raw)
df_raw.set_index('Action', inplace=True)
df_raw.sort_values(by='Proj_id',ascending=True,inplace=True)
df_raw_copy.sort_values(by='Proj_id',ascending=False,inplace=True)


# To calculate 1 Year Rolling NSV/iNSV(s)
def calculate_rolling_values(row, doll_year, doll_wrap):
    if row['launch_yr'] == row['yr_of_nsv']:
        return row[doll_year]
    elif row['launch_yr'] == row['yr_of_nsv'] - 1:
        return row[doll_wrap]
    else:
        return 0

def weighted_average_of_group(df,dim_selected):

    df['Weighted_Score1'] = df['old_gross_mrgn_pct'] * df['old_nsv_yr_1']
    df['Weighted_Score2'] = df['new_gross_mrgn_pct'] * df['new_nsv_yr_1']

    weighted_avg_df = df.groupby(dim_selected).apply(lambda x: pd.Series({
        'old_Weighted_Average_Score': x['Weighted_Score1'].sum() / x['old_nsv_yr_1'].sum(),
        'new_Weighted_Average_Score': x['Weighted_Score2'].sum() / x['new_nsv_yr_1'].sum()
    })).reset_index()
    return (weighted_avg_df)

def remove_decimal(number):
    if number.endswith('.0'):
        return number[:-2]
    else:
        return number

def Process_data(df, dfx):
    # Rename columns
    dfx.rename(columns={'fltr': 'filter'}, inplace=True)
    
    # Replace values based on condition
    columns_to_replace = ['new_nsv_yr_1', 'new_nsv_yr_2', 'new_nsv_yr_3', 
                          'new_nsv_yr_1_rollup', 'new_nsv_yr_3_rollup', 
                          'new_nsv_year', 'new_nsv_wrap', 
                          'new_insv_yr_1', 'new_insv_yr_2', 'new_insv_yr_3', 
                          'new_insv_yr_1_rollup', 'new_insv_yr_3_rollup', 
                          'new_insv_year', 'new_insv_wrap', 
                          'new_canblz_pct', 'new_gross_mrgn_pct', 
                          'new_rd_days',
                          'new_gsv_yr_1','new_gsv_yr_1','new_gsv_yr_3','new_gsv_yr_1_rollup',
                          'new_gsv_yr_3_rollup','new_gsv_year','new_gsv_wrap']
    for column in columns_to_replace:
        df[column] = [row[column] if row['flag'] else None for i, row in df.iterrows()]
    
    # Convert 'yr_of_nsv' to string and remove decimal
    dfx['new_yr_of_nsv'] = dfx['yr_of_nsv'].astype(str).apply(remove_decimal)
    df['new_yr_of_nsv'] = df['yr_of_nsv'].astype(str).apply(remove_decimal)
    dfx['gross_mrgn_pct'] = dfx['gross_mrgn_pct'].fillna(0)
    df['gross_mrgn_pct'] = df['new_gross_mrgn_pct'].fillna(0)
    dfx['gross_mrgn_pct'] = dfx['gross_mrgn_pct'].astype('int64')
    df['gross_mrgn_pct'] = df['gross_mrgn_pct'].astype('int64')
    
    # Create 'Combined' column for joining
    df['Combined'] = df['new_yr_of_nsv'] + "_" + df['Project']
    dfx['Combined'] = dfx['new_yr_of_nsv'] + "_" + dfx['Project']
    
    # Join DataFrames
    result_concat = df.join(dfx.set_index("Combined"), how='left', rsuffix="_df1", on="Combined")
    
    # Drop unnecessary columns and rename remaining columns
    columns_to_drop = ['Project_df1', 'Region_df1', 'Proj_id_df1', 'yr_of_nsv_df1', 'launch_dt_df1', 
                       'launch_yr_df1', 'new_yr_of_nsv', 'new_yr_of_nsv_df1']
    result_concat.drop(columns_to_drop, axis=1, inplace=True)
    result_concat.rename(columns={'nsv_yr': 'old_nsv_year', 'nsv_wrap': 'old_nsv_wrap', 
                                  'nsv_yr_1': 'old_nsv_yr_1', 'nsv_yr_2': 'old_nsv_yr_2', 
                                  'nsv_yr_3': 'old_nsv_yr_3', 'nsv_yr_1_rollup': 'old_nsv_yr_1_rollup', 
                                  'nsv_yr_3_rollup': 'old_nsv_yr_3_rollup', 'insv_yr': 'old_insv_year', 
                                  'insv_wrap': 'old_insv_wrap', 'insv_yr_1': 'old_insv_yr_1', 
                                  'insv_yr_2': 'old_insv_yr_2', 'insv_yr_3': 'old_insv_yr_3', 
                                  'insv_yr_1_rollup': 'old_insv_yr_1_rollup', 'insv_yr_3_rollup': 'old_insv_yr_3_rollup', 
                                  'rd_days': 'old_rd_days', 'canblz_pct': 'old_canblz_pct', 
                                  'gross_mrgn_pct': 'old_gross_mrgn_pct',
                                  'gsv_yr_1':'old_gsv_yr_1','gsv_yr_2':'old_gsv_yr_2','gsv_yr_3':'old_gsv_yr_3',
                                  'gsv_yr_1_rollup':'old_gsv_yr_1_rollup','gsv_yr_3_rollup':'old_gsv_yr_3_rollup', 
                                  'gsv_yr':'old_gsv_year','gsv_wrap':'old_gsv_wrap'
                                  }, inplace=True)
    # Adding 3 year rolling NSV 
    result_concat['new_3_year_rolling_nsv']= result_concat['new_nsv_year'].fillna(0)+ result_concat['new_nsv_wrap'].fillna(0)
    result_concat['old_3_year_rolling_nsv']= result_concat['old_nsv_year'].fillna(0)+ result_concat['old_nsv_wrap'].fillna(0)
    result_concat['new_3_year_rolling_nsv'] = [None if row['flag'] ==False else row['new_3_year_rolling_nsv'] for index, row in result_concat.iterrows()]
    # Adding 3 year rolling iNSV 
    result_concat['new_3_year_rolling_insv']= result_concat['new_insv_year'].fillna(0)+ result_concat['new_insv_wrap'].fillna(0)
    result_concat['old_3_year_rolling_insv']= result_concat['old_insv_year'].fillna(0)+ result_concat['old_insv_wrap'].fillna(0)
    result_concat['new_3_year_rolling_insv'] = [None if row['flag'] ==False else row['new_3_year_rolling_insv'] for index, row in result_concat.iterrows()]

    result_concat['new_3_year_rolling_gsv']= result_concat['new_gsv_year'].fillna(0)+ result_concat['new_gsv_wrap'].fillna(0)
    result_concat['old_3_year_rolling_gsv']= result_concat['old_gsv_year'].fillna(0)+ result_concat['old_gsv_wrap'].fillna(0)
    result_concat['new_3_year_rolling_gsv'] = [None if row['flag'] ==False else row['new_3_year_rolling_gsv'] for index, row in result_concat.iterrows()]

    # Calculate rolling NSV
    result_concat['old_1_year_rolling_nsv'] = result_concat.apply(lambda row: calculate_rolling_values(row, 'old_nsv_year', 'old_nsv_wrap'), axis=1)
    result_concat['new_1_year_rolling_nsv'] = result_concat.apply(lambda row: calculate_rolling_values(row, 'new_nsv_year', 'new_nsv_wrap'), axis=1)
    result_concat['new_1_year_rolling_nsv'] = result_concat['new_1_year_rolling_nsv'].where(result_concat['flag'], None)

    # Calculate rolling iNSV
    result_concat['old_1_year_rolling_insv'] = result_concat.apply(lambda row: calculate_rolling_values(row, 'old_insv_year', 'old_insv_wrap'), axis=1)
    result_concat['new_1_year_rolling_insv'] = result_concat.apply(lambda row: calculate_rolling_values(row, 'new_insv_year', 'new_insv_wrap'), axis=1)
    result_concat['new_1_year_rolling_insv'] = result_concat['new_1_year_rolling_insv'].where(result_concat['flag'], None)
    
    #calculate rolling GSV
    result_concat['old_1_year_rolling_gsv'] = result_concat.apply(lambda row: calculate_rolling_values(row, 'old_gsv_year', 'old_gsv_wrap'), axis=1)
    result_concat['new_1_year_rolling_gsv'] = result_concat.apply(lambda row: calculate_rolling_values(row, 'new_gsv_year', 'new_gsv_wrap'), axis=1)
    result_concat['new_1_year_rolling_gsv'] = result_concat['new_1_year_rolling_gsv'].where(result_concat['flag'], None)

    return result_concat

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
                    df ) unpivot (mm_nsv for mm_type in (mm_nsv_yr_1, mm_nsv_yr_2, mm_nsv_yr_3, mm_nsv_yr_dummy))) ),

			insv_calc as (
            select * ,
            row_number() over(partition by Project order by year_of_nsv) filtr ,
            lag(mm_insv) over (partition by Project order by year_of_nsv ) next_mm_insv
            from (
            select rd_days,bu,proj_desc,porfolio_bucket,pd_or_ft_days,pkg_days,canblz_pct,gross_mrgn_pct
                ,flag,upper(Action) as Action,Project,Proj_id,launch_dt,Region,insv_yr_1,insv_yr_2,insv_yr_3,
                launch_yr + launch_month_nov_dec_flag launch_year ,
                case
                    when launch_month_nov_dec_flag = 1 then 1
                    else launch_month
                end launch_month ,
                launch_yr + launch_month_nov_dec_flag +
                case
                    when mm_type = 'mm_insv_yr_1' then 0
                    when mm_type = 'mm_insv_yr_2' then 1
                    when mm_type = 'mm_insv_yr_3' then 2
                    when mm_type = 'mm_insv_yr_dummy' then 3
                    else 0
                end year_of_nsv ,
                mm_insv *(13-launch_month) yearly_total ,
                mm_insv *(launch_month-1) mm_insv ,
                insv_yr_1_rollup ,
                insv_yr_3_rollup ,
                curr_yr
            from (             
                select rd_days,bu,proj_desc,porfolio_bucket,pd_or_ft_days,pkg_days,canblz_pct,gross_mrgn_pct
                    ,flag,upper(Action) as Action ,Project,Proj_id,launch_dt,Region,insv_yr_1,insv_yr_2,insv_yr_3,
                    extract( year from cast(launch_dt as date)) as launch_yr,
                    extract( month from cast(launch_dt as date)) as launch_month ,
                    case when launch_month in (11, 12) then 1 else 0 end launch_month_nov_dec_flag ,
                    cast(coalesce(insv_yr_1, 0) as decimal)/ 12 as mm_insv_yr_1 ,
                    cast(coalesce(insv_yr_2, 0) as decimal)/ 12 as mm_insv_yr_2 ,
                    cast(coalesce(insv_yr_3, 0) as decimal)/ 12 as mm_insv_yr_3 ,
                    0 as mm_insv_yr_dummy ,
                    cast(coalesce(insv_yr_1, 0) as decimal) insv_yr_1_rollup ,
                    cast(coalesce(insv_yr_1, 0) as decimal) + cast(coalesce(insv_yr_2, 0) as decimal) + cast(coalesce(insv_yr_3, 0) as decimal) insv_yr_3_rollup,
                    extract(year from current_date) curr_yr
                from
                    df ) unpivot (mm_insv for mm_type in (mm_insv_yr_1, mm_insv_yr_2, mm_insv_yr_3, mm_insv_yr_dummy))) ),
                         
            gsv_calc as (
            select * ,
            row_number() over(partition by Project order by year_of_nsv) filtr ,
            lag(mm_gsv) over (partition by Project order by year_of_nsv ) next_mm_gsv
            from (
            select flag,upper(Action) as Action,Project,Proj_id,launch_dt,Region,gsv_yr_1,gsv_yr_2,gsv_yr_3,
                launch_yr + launch_month_nov_dec_flag launch_year ,
                case
                    when launch_month_nov_dec_flag = 1 then 1
                    else launch_month
                end launch_month ,
                launch_yr + launch_month_nov_dec_flag +
                case
                    when mm_type = 'mm_gsv_yr_1' then 0
                    when mm_type = 'mm_gsv_yr_2' then 1
                    when mm_type = 'mm_gsv_yr_3' then 2
                    when mm_type = 'mm_gsv_yr_dummy' then 3
                    else 0
                end year_of_nsv ,
                mm_gsv *(13-launch_month) yearly_total ,
                mm_gsv *(launch_month-1) mm_gsv ,
                gsv_yr_1_rollup ,
                gsv_yr_3_rollup ,
                curr_yr
            from (             
                select flag,upper(Action) as Action ,Project,Proj_id,launch_dt,Region,gsv_yr_1,gsv_yr_2,gsv_yr_3,
                    extract( year from cast(launch_dt as date)) as launch_yr,
                    extract( month from cast(launch_dt as date)) as launch_month ,
                    case when launch_month in (11, 12) then 1 else 0 end launch_month_nov_dec_flag ,
                    cast(coalesce(gsv_yr_1, 0) as decimal)/ 12 as mm_gsv_yr_1 ,
                    cast(coalesce(gsv_yr_2, 0) as decimal)/ 12 as mm_gsv_yr_2 ,
                    cast(coalesce(gsv_yr_3, 0) as decimal)/ 12 as mm_gsv_yr_3 ,
                    0 as mm_gsv_yr_dummy ,
                    cast(coalesce(gsv_yr_1, 0) as decimal) gsv_yr_1_rollup ,
                    cast(coalesce(gsv_yr_1, 0) as decimal) + cast(coalesce(gsv_yr_2, 0) as decimal) + cast(coalesce(gsv_yr_3, 0) as decimal) gsv_yr_3_rollup,
                    extract(year from current_date) curr_yr
                from
                    df ) unpivot (mm_gsv for mm_type in (mm_gsv_yr_1, mm_gsv_yr_2, mm_gsv_yr_3, mm_gsv_yr_dummy))) )             

				select insv.flag flag, upper(insv.Action) as Action ,nsv.Project Project,
				nsv.launch_dt launch_dt
				,nsv.Region Region
				,nsv.Proj_id Proj_id
				,nsv.launch_year launch_yr
				,nsv.launch_month launch_mth
				,nsv.year_of_nsv yr_of_nsv
				,nsv.filtr fltr
				,nsv.nsv_yr_1
				,nsv.nsv_yr_2
				,nsv.nsv_yr_3,
				insv.insv_yr_1,
				insv.insv_yr_2,
				insv.insv_yr_3,
				insv.mm_insv,
				insv.yearly_total as insv_year,
				insv.next_mm_insv as insv_wrap,
                insv.insv_yr_1_rollup,
                insv.insv_yr_3_rollup
				,nsv.yearly_total as nsv_year
				,nsv.next_mm_nsv as nsv_wrap 
				,nsv.nsv_yr_1_rollup
				,nsv.nsv_yr_3_rollup
                ,insv.rd_days, insv.bu, insv.proj_desc, insv.porfolio_bucket, insv.pd_or_ft_days, insv.pkg_days, insv.canblz_pct, insv.gross_mrgn_pct
                ,gsv.yearly_total as gsv_year
                ,gsv.next_mm_gsv as gsv_wrap 
                ,gsv.gsv_yr_1_rollup gsv_yr_1_rollup
                ,gsv.gsv_yr_3_rollup gsv_yr_3_rollup
                ,gsv.gsv_yr_1
                ,gsv.gsv_yr_2
                ,gsv.gsv_yr_3

                from nsv_calc nsv left join insv_calc insv         
                on nsv.Project=insv.Project and nsv.year_of_nsv=insv.year_of_nsv
                left join gsv_calc gsv on nsv.Project=gsv.Project and nsv.year_of_nsv = gsv.year_of_nsv
            

            )
             """
            ).df()

    #Renaming and dropping few of the columns 
    z_df1.drop(['launch_mth'],axis=1,inplace=True)
    z_df1.rename(columns={'year_of_nsv':'yr_of_nsv', 'launch_year':'launch_yr',
                          'nsv_year':'new_nsv_year','insv_year':'new_insv_year','filtr':'filter','nsv_wrap':'new_nsv_wrap',
                          'insv_wrap':'new_insv_wrap','insv_yr_1': 'new_insv_yr_1','insv_yr_2': 'new_insv_yr_2','insv_yr_3': 'new_insv_yr_3','insv_yr_1_rollup': 'new_insv_yr_1_rollup','insv_yr_3_rollup': 'new_insv_yr_3_rollup',
                          'nsv_yr_1': 'new_nsv_yr_1','nsv_yr_2': 'new_nsv_yr_2','nsv_yr_3': 'new_nsv_yr_3','nsv_yr_1_rollup': 'new_nsv_yr_1_rollup','nsv_yr_3_rollup': 'new_nsv_yr_3_rollup','rd_days':'new_rd_days','canblz_pct':'new_canblz_pct',
                          'gross_mrgn_pct':'new_gross_mrgn_pct',
                          'gsv_yr_1':'new_gsv_yr_1','gsv_yr_2':'new_gsv_yr_2','gsv_yr_3':'new_gsv_yr_3','gsv_yr_3_rollup':'new_gsv_yr_3_rollup','gsv_yr_1_rollup':'new_gsv_yr_1_rollup','gsv_wrap':'new_gsv_wrap','gsv_year':'new_gsv_year'}, inplace=True)
    return(z_df1)

def plot_bar(df,measure,dim_selected):
    # for measures
    if measure == "1 year rolling nsv":
        old_column = 'old_1_year_rolling_nsv'
        new_column = 'new_1_year_rolling_nsv'
        ylabel = '1 year nsv'
    elif measure == "1 year rolling insv":
        old_column = 'old_1_year_rolling_insv'
        new_column = 'new_1_year_rolling_insv'
        ylabel = '1 year insv'
    elif measure == "3 year rolling nsv":
        old_column = 'old_3_year_rolling_nsv'
        new_column = 'new_3_year_rolling_nsv'
        ylabel = '3 year nsv'
    elif measure == "3 year rolling insv":
        old_column = 'old_3_year_rolling_insv'
        new_column = 'new_3_year_rolling_insv'
        ylabel = '3 year insv'
    elif measure == "R&D Days":
        old_column = 'old_rd_days'
        new_column = 'new_rd_days'
        ylabel = '3 year r&d_days'
    elif measure == "1 year rolling gsv":
        old_column = 'old_1_year_rolling_gsv'
        new_column = 'new_1_year_rolling_gsv'
        ylabel = '3 year gsv'
    elif measure == "3 year rolling gsv":
        old_column = 'old_3_year_rolling_gsv'
        new_column = 'new_3_year_rolling_gsv'
        ylabel = '3 year gsv'
    
    df[old_column]=df[old_column].round(2)
    df[new_column]=df[new_column].round(2)
    if measure == "R&D Days":
        grp_by = df.groupby([dim_selected])[[old_column,new_column]].sum().reset_index()
        grp_by[old_column]=grp_by[old_column]/4
        grp_by[new_column]=grp_by[new_column]/4
    else :
        grp_by = df.groupby(dim_selected)[[old_column,new_column]].sum().reset_index()
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    # Width of the bars
    bar_width = 0.2
    # Index for the x-axis
    ind = range(len(grp_by))
    # Plotting old sales
    old_sales = ax.bar(ind, grp_by[old_column], bar_width, label='Old Sales')
    new_sales = ax.bar([i + bar_width for i in ind], grp_by[new_column], bar_width, label='New Sales')
    # Setting labels and title
    ax.set_xlabel(dim_selected)
    ax.set_ylabel(ylabel)
    ax.set_xticks([i + bar_width / 2 for i in ind])
    ax.set_xticklabels(grp_by[dim_selected])
    #ax.set(ylim=(10, 150))
    ax.legend()
    # Show plot
    st.pyplot()
    grp_by['Difference'] = grp_by[new_column] - grp_by[old_column]
    st.dataframe(grp_by,height=175)

def plot_gm(df,dim_selected):
    gm = weighted_average_of_group(df,dim_selected)
    ylabel = 'gross margin'    
    gm['old_Weighted_Average_Score']=gm['old_Weighted_Average_Score'].round(2)
    gm['new_Weighted_Average_Score']=gm['new_Weighted_Average_Score'].round(2)
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    # Width of the bars
    bar_width = 0.2
    # Index for the x-axis
    ind = range(len(gm))
    # Plotting old sales
    old_sales = ax.bar(ind,gm['old_Weighted_Average_Score'], bar_width, label='Old')
    new_sales = ax.bar([i + bar_width for i in ind], gm['new_Weighted_Average_Score'], bar_width, label='New')
    # Setting labels and title
    ax.set_xlabel(dim_selected)
    ax.set_ylabel(ylabel)
    ax.set_xticks([i + bar_width / 2 for i in ind])
    ax.set_xticklabels(gm[dim_selected])
    #ax.set(ylim=(10, 150))
    ax.legend()
    # Show plot
    st.pyplot()
    gm['Difference'] = gm['new_Weighted_Average_Score'] - gm['old_Weighted_Average_Score']
    st.dataframe(gm,height=175)
    
def validate_main(df):
    # duplicate project names 
    errors = []
    #Validate column names
    expected_columns = ['Action', 'Proj_id','launch_yr','rd_days','Region','bu','Project','proj_desc','porfolio_bucket',
                        'pd_or_ft_days','pkg_days','canblz_pct','gross_mrgn_pct','nsv_yr_1','nsv_yr_2','nsv_yr_3',
                        'insv_yr_1','insv_yr_2','insv_yr_3','launch_dt','gsv_yr_1','gsv_yr_2','gsv_yr_3']
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
    if df['insv_yr_1'].dtype == 'object' or df['insv_yr_2'].dtype == 'object' or df['insv_yr_3'].dtype == 'object' :
        errors.append(f"Error: There are non-numerical values in iNSV columns.\n\n  \t\t Please provide the numerical values.")
    if df['gsv_yr_1'].dtype == 'object' or df['gsv_yr_2'].dtype == 'object' or df['gsv_yr_3'].dtype == 'object' :
        errors.append(f"Error: There are non-numerical values in GSV columns.\n\n  \t\t Please provide the numerical values.")
    
    return errors

def validate_sub(df):
    errors = []
    columns_to_check = ['Region','bu','proj_desc','porfolio_bucket']
    for column in columns_to_check:
        if df[column].isnull().any():
            errors.append(f"Warning: There are null values in column '{column}'.    Please provide the values for better result.")
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

        # filtering the ADD rows only
        df1 = df1[df1['Action']=='Add']
        df_concat = pd.concat([df1,df_raw_copy])

        if len(validate_main(df_concat))>0:
            for i in validate_main(df_concat):
                st.error(i)
        else :
            # validation
            for i in validate_sub(df_concat):
                st.warning(i)
            # changing the data type and format of launch date
            df_concat['launch_dt'] = pd.to_datetime(df_concat['launch_dt'])
            df_concat['launch_dt'] = df_concat['launch_dt'].dt.strftime('%Y-%m-%d')
            user_input_data = copy.deepcopy(df_concat)
            dynamic_filters = DynamicFilters(df_concat, filters=['Project', 'Region'])
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

                # Processing the data 
                sql_result = sql_process(q)
                final = Process_data(sql_result,raw_copy)
                final.sort_values(by='Proj_id',ascending=False,inplace=True)
                final.drop(['flag','gm_pct_yr_1','gm_pct_yr_2','gm_pct_yr_3','file_nm','kortex_upld_ts'],axis=1,inplace=True)

                # Dropdown for the measure
                measure_selected = st.selectbox("Select the Measure: ",
                                                ('1 year rolling nsv','1 year rolling insv','1 year rolling gsv','3 year rolling nsv','3 year rolling insv','3 year rolling gsv',
                                                 'R&D Days','Gross Margin'))
                
                # Ploting the Region and Year of nsv
                col1,col2=st.columns(2) 
                with col1:
                    if measure_selected == 'Gross Margin':
                        plot_gm(final,'Region')
                    else :
                        st.subheader('Old vs new by Region')
                        plot_bar(final,measure_selected,'Region')
                with col2:
                    if measure_selected =='R&D Days' or  measure_selected =='Gross Margin':
                        st.text("The selected metric is not available at Year of NSV") 
                    else : 
                        st.subheader('Old vs new by Year of NSV')
                        plot_bar(final,measure_selected,'yr_of_nsv')

                # Ploting the Launch year and bu 
                col1,col2=st.columns(2) 
                with col1:
                    if measure_selected == 'Gross Margin':
                        plot_gm(final,'bu')
                    else :
                        st.subheader('Old vs new by BU')
                        plot_bar(final,measure_selected,'bu')
                with col2:
                    if measure_selected == 'Gross Margin':
                        plot_gm(final,'launch_yr')
                    else :
                        st.subheader('Old vs new by Launch Year')
                        plot_bar(final,measure_selected,'launch_yr')

                 # Ploting the Porfolio bucket
                col1,col2=st.columns(2) 
                with col1:
                    if measure_selected == 'Gross Margin':
                        plot_gm(final,'porfolio_bucket')
                    else :
                        st.subheader('Old vs new by porfolio bucket')
                        plot_bar(final,measure_selected,'porfolio_bucket')
                with col2:
                    st.write() 
                    
                
                st.write('The Processed Data',final)
    

if __name__ == "__main__":
    main()
