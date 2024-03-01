import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page Settings
st.set_page_config(page_title='Kellogg Dynamic POC Simulator',page_icon=':smile:')

# CSS
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

def calculate_totals(df1, df2):
    # Concatenate df1 and df2
    combined_df = pd.concat([df1, df2])

    # Calculate total NSV for df1
    total_nsv_df1 = df1['NSV'].sum()

    # Calculate total NSV for combined df1 and df2
    total_nsv_combined = combined_df['NSV'].sum()

    return total_nsv_df1, total_nsv_combined

def plot_totals(total_nsv_df1, total_nsv_combined, selected_projects, df1, df2):
    # Plotting
    fig, ax = plt.subplots()

    if not selected_projects:
        # Plot bars for combined unfiltered values
        ax.bar('Old NSV', total_nsv_df1, color='#003f5c', label='Old NSV')
        ax.bar('New NSV', total_nsv_combined, color='#ff6361', label='New NSV')
    else:
        for project in selected_projects:
            if project:
                # Calculate total NSV for the selected project
                total_nsv_selected_project_df1 = df1[df1['Project Name'] == project]['NSV'].sum()
                total_nsv_selected_project_df2 = df2[df2['Project Name'] == project]['NSV'].sum()
                total_nsv_selected_project_combined = total_nsv_selected_project_df1 + total_nsv_selected_project_df2

                # Plot bars for selected project
                ax.bar('Old NSV', total_nsv_selected_project_df1, color='#003f5c', label=f'Total NSV for project {project} in df1')
                ax.bar('New NSV', total_nsv_selected_project_combined, color='#ff6361', label=f'Total NSV for project {project} in combined df1 and df2')

    ax.set_ylabel('Total NSV ($)')
    st.pyplot(fig)

def main():
    st.title('Kellogg Dynamic POC Simulator')

    # Create df1
    df1_data = {'Project Name': ['A', 'B', 'C'],
                'Country': ['India', 'India', 'US'],
                'NSV': [100, 200, 300]}
    df1 = pd.DataFrame(df1_data)

    # Sidebar Excel upload for df2
    st.sidebar.subheader('Upload simulation Excel file')
    df2_uploaded_file = st.sidebar.file_uploader('Upload Excel', type=['xlsx', 'xls'])
    if df2_uploaded_file is not None:
        df2 = pd.read_excel(df2_uploaded_file)

        total_nsv_df1, total_nsv_combined = calculate_totals(df1, df2)

        # Multiselect dropdown filter for project names
        all_projects = sorted(list(set(df1['Project Name'].unique()) | set(df2['Project Name'].unique())))
        selected_projects = st.multiselect('Select Project Name(s)', all_projects)

        st.subheader('NSV Comparison')
        plot_totals(total_nsv_df1, total_nsv_combined, selected_projects, df1, df2)
    else:
        st.warning('Please upload the simulation Excel file.')

if __name__ == "__main__":
    main()
