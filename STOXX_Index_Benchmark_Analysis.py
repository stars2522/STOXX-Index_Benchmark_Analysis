import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import functions_file as functions

# Define the function to process the data
def process_index_data(file_path, portfolio_name):
    # Read the CSV file
    index_file = pd.read_csv(file_path, delimiter=';')
    
    # Select relevant columns and convert 'Date' to datetime format
    index_file2 = index_file[['Date', 'Indexvalue']]
    index_file2['Date'] = pd.to_datetime(index_file2['Date'])
    
    # Rename the columns based on the provided portfolio name
    index_file2.rename(columns={
        'Indexvalue': f'Indexvalue_{portfolio_name}'
    }, inplace=True)

    # Find the last date in the dataset
    last_date = index_file2['Date'].max()
    
    # Identify the last available month and year
    last_month = last_date.month
    last_year = last_date.year
    
    # Identify the next month (next_month will be the current month + 1)
    next_month = last_month + 1 if last_month < 12 else 1
    next_year = last_year if last_month < 12 else last_year + 1
    
    # Check if there is any data for the next month (this would indicate if the current month is incomplete)
    next_month_data = index_file2[(index_file2['Date'].dt.month == next_month) & (index_file2['Date'].dt.year == next_year)]
    
    # If data for the next month exists, we consider the current month as incomplete
    if len(next_month_data) == 0:  # No data for next month means the current month is incomplete
        incomplete_month = last_month
        # st.write(f"Considering {last_month}/{last_year} is incomplete, analysis will be till ")
    else:
        incomplete_month = None
        # st.write(f"Next month's data exists, considering {last_month}/{last_year} as a complete month.")
    
    # Remove the data for the incomplete month if necessary
    if incomplete_month:
        index_file2 = index_file2[~((index_file2['Date'].dt.month == incomplete_month) & (index_file2['Date'].dt.year == last_year))]

    # Calculate the return and cumulative return on the remaining data
    index_file2[f'Return_{portfolio_name}'] = index_file2[f'Indexvalue_{portfolio_name}'].pct_change()
    index_file2[f'Cumulative_return_{portfolio_name}'] = (1 + index_file2[f'Return_{portfolio_name}']).cumprod() - 1
    
    return index_file2


# Adjust the page configuration to have more space for charts
st.set_page_config(
    page_title="Streamlit App",
    page_icon="📊",
    layout="wide",  # "wide" layout increases the width of the content area
    initial_sidebar_state="expanded"  # Expands the sidebar by default
)
# Streamlit app
def main():
    # Sidebar for user input
    st.sidebar.title('Index-Benchmark Level Analysis')
    
    index_file = st.sidebar.file_uploader("Upload the Index file (CSV or TXT)", type=["csv", "txt"])
    benchmark_file = st.sidebar.file_uploader("Upload the Benchmark file (CSV or TXT)", type=["csv", "txt"])
    
    index_name = st.sidebar.text_input("Enter the Index name (CAPITAL LETTERS, e.g., 'SAXPMFGR')")
    benchmark_name = st.sidebar.text_input("Enter the Benchmark name (CAPITAL LETTERS, e.g., 'SXXGR')")
    
    # Main area
    st.title('Index-Benchmark Level Analysis')
    st.write("""Upload your index file and benchmark file, and the app will process them and display their Index levels, Cumulative returns, Risk-Return Profile, Annual Returns.""")

    # Proceed with processing if both files are uploaded and names are provided
    if index_file and benchmark_file and index_name and benchmark_name:
        # Process index and benchmark data
        index_data = process_index_data(index_file, index_name)
        benchmark_data = process_index_data(benchmark_file, benchmark_name)
        
        # Merge both dataframes based on Date
        merged_data = pd.merge(index_data, benchmark_data, on='Date', how='inner')
### Addtions 2
        annual_returns_df = functions.calculate_annual_returns(merged_data, index_name, benchmark_name)
        annual_returns_df2=annual_returns_df[['Year',f'Annual_return_{index_name}',f'Annual_return_{benchmark_name}']]
        annual_returns_df2.rename(columns={f'Annual_return_{index_name}':f'{index_name}',f'Annual_return_{benchmark_name}':f'{benchmark_name}'},inplace=True)
        annual_returns_df2[f'{index_name}'] = annual_returns_df2[f'{index_name}'] * 100
        annual_returns_df2[f'{benchmark_name}'] = annual_returns_df2[f'{benchmark_name}'] * 100

# Format the columns to include percentage signs
        annual_returns_df2[f'{index_name}'] = annual_returns_df2[f'{index_name}'].apply(lambda x: f'{x:.2f}%')
        annual_returns_df2[f'{benchmark_name}'] = annual_returns_df2[f'{benchmark_name}'].apply(lambda x: f'{x:.2f}%')
        annual_returns_df2.set_index('Year',inplace=True)
        # st.dataframe(annual_returns_df.tail(8)) 
#### ADDITIONS
        common_dates = index_data['Date'].unique().tolist()
        benchmark_data = benchmark_data[benchmark_data['Date'].isin(common_dates)]
        # Calculate the period dates and RNR values
        period_dates_index = functions.get_period_dates(index_data)
        period_rnr_index = functions.get_rnr_values(index_data, period_dates_index, index_name)

        period_dates_benchmark = functions.get_period_dates(benchmark_data)
        period_rnr_benchmark = functions.get_rnr_values(benchmark_data, period_dates_benchmark, benchmark_name)

        # Merge RNR data for both index and benchmark
        rnr_benchmark_index = pd.merge(period_rnr_index, period_rnr_benchmark, 
                                       on=['Period', 'Start Date', 'End Date'])
        # Check if data is present in merged DataFrame
        # st.write(rnr_benchmark_index.head())  # Debugging line to ensure data is correct
        rnr_benchmark_index2=rnr_benchmark_index[['Period',f'Annualised Return_{index_name}',f'Annualised Return_{benchmark_name}',f'Volatility_{index_name}',f'Volatility_{benchmark_name}']]
        columns_to_process=[f'Annualised Return_{index_name}',f'Annualised Return_{benchmark_name}',f'Volatility_{index_name}',f'Volatility_{benchmark_name}']
        for col in columns_to_process:
            rnr_benchmark_index2[col] = rnr_benchmark_index2[col] * 100
            rnr_benchmark_index2[col] = rnr_benchmark_index2[col].apply(lambda x: f'{x:.2f}%')
        rnr_benchmark_index2.set_index('Period',inplace=True)

        # print(px.colors.qualitative.Plotly)
        # Plotting the Index Level chart using Plotly
        fig_index_level = px.line(merged_data, x='Date', y=[f'Indexvalue_{index_name}', f'Indexvalue_{benchmark_name}'],
                                  title='Index Levels',
                                  labels={'Date': 'Date', 
                                          f'{index_name}': f'Index Level: {index_name}', 
                                          f'{benchmark_name}': f'Index Level: {benchmark_name}'})
        # Set the label for the y-axis
        fig_index_level.update_layout(
                   yaxis_title="Index Level",  # You can adjust this label as per your need
                      )
        # Inspecting the color of the lines used in the figure
        for trace in fig_index_level['data']:
            print(f"Trace Name: {trace['name']}, Line Color: {trace['line']['color']}")

        # Plotting the Cumulative Return chart using Plotly
        fig_cumulative_return = px.line(merged_data, x='Date', y=[f'Cumulative_return_{index_name}', f'Cumulative_return_{benchmark_name}'],
                                        title='Cumulative Returns',
                                        labels={'Date': 'Date', 
                                                f'Cumulative_return_{index_name}': f'Cumulative Return: {index_name}', 
                                                f'Cumulative_return_{benchmark_name}': f'Cumulative Return: {benchmark_name}'})
        fig_cumulative_return.update_layout(
                   yaxis_title="Cumulative returns",yaxis=dict(tickformat='.0%'),  # You can adjust this label as per your need
                      )
        # Plot the first chart: Index and Benchmark Returns
        fig_rnr = go.Figure()

        # Add bar traces for the Annualized Returns of index and benchmark
        fig_rnr.add_trace(go.Bar(
            x=[i for i in range(len(rnr_benchmark_index))],
            y=rnr_benchmark_index[f'Annualised Return_{index_name}'],
            name=f'{index_name} Return',
            marker_color='#000001',
            width=0.4
        ))

        fig_rnr.add_trace(go.Bar(
            x=[i + 0.4 for i in range(len(rnr_benchmark_index))],
            y=rnr_benchmark_index[f'Annualised Return_{benchmark_name}'],
            name=f'{benchmark_name} Return',
            marker_color='#000002',
            width=0.4
        ))
        #Add scatter traces for the Volatility of index and benchmark on secondary y-axis
        fig_rnr.add_trace(go.Scatter(
            x=[i for i in range(len(rnr_benchmark_index))],
            y=rnr_benchmark_index[f'Volatility_{index_name}'],
            name=f'{index_name} Volatility',
            mode='markers',
            marker=dict(color='darkblue', size=10, line=dict(width=2, color='red')),
            yaxis='y2',
            marker_color='#000001'
        ))
        fig_rnr.add_trace(go.Scatter(
            x=[i + 0.4 for i in range(len(rnr_benchmark_index))],
            y=rnr_benchmark_index[f'Volatility_{benchmark_name}'],
            name=f'{benchmark_name} Volatility',
            mode='markers',
            marker=dict(color='darkblue', size=10, line=dict(width=2, color='red')),
            yaxis='y2',
            marker_color='#000002'
        ))
        # Update layout
        fig_rnr.update_layout(
            title="Risk-Return Profile (Different Periods)",
            xaxis_title="Period",
            yaxis_title="Annualised Return",
            yaxis=dict(tickformat='.0%',showgrid=True),
            yaxis2=dict(
                title="Volatility",
                overlaying="y",
                side="right",
                tickformat='.0%',
                showgrid=False
            ),
            xaxis=dict(
            tickmode='array',
            tickvals=[i + 0.2 for i in range(len(rnr_benchmark_index))],  # Adjust position for ticks
            ticktext=rnr_benchmark_index['Period'],  # Ensure Period is displayed as tick labels
            tickcolor='navy'  # Change tick color
            ),
            barmode='group',
            template="plotly_dark",
            showlegend=True,
            legend=dict(
              x=1.08,  # Position legend to the right of the plot (adjust as needed)
              y=1,  # Position legend at the top
              traceorder='normal',
              orientation='v'  # Vertical orientation
    ),
        )
        # Create the Plotly figure
        fig_ann_ret = go.Figure()

# Add Bar plot for SAXPMFGR
        fig_ann_ret.add_trace(go.Bar(
        x=annual_returns_df['Year'],  # X-axis is the Year column
        y=annual_returns_df[f'Annual_return_{index_name}'],  # Y-axis is the Annual Return for SAXPMFGR
        name=f'{index_name}',
        marker={'color': '#000001'}
        ))

        fig_ann_ret.add_trace(go.Bar(
       x=annual_returns_df['Year'],  # X-axis is the Year column
       y=annual_returns_df[f'Annual_return_{benchmark_name}'],  # Y-axis is the Annual Return for SXXGR
       name=f'{benchmark_name}',
       marker={'color': '#000002'}
       ))

        # Update layout with custom settings
        fig_ann_ret.update_layout(
        title="Annual Returns",
        barmode='group',  # This makes the bars appear side by side
        xaxis=dict(
        title='Year',
        showgrid=False,
        # zeroline=True,  # Optional: Remove the x-axis zero line
        ticks='outside',  # Adds tick marks on the outside of the axis
        # tickson='boundaries',
        dtick=1  # Tick marks at the boundaries, which includes zero
    ),
        yaxis=dict(
        title='Annual Return',
        tickformat='.0%',
        # tickfont={'color': 'navy'},
        showgrid=True,
        # gridcolor='lightblue',
        gridwidth=1,
        # zerolinecolor='navy',
        # zeroline=True,  # Optional: Remove the y-axis zero line
        # showline=True,  # Optional: Remove the y-axis line
        ticks='outside',  # Adds tick marks on the outside of the axis
        # tickson='boundaries',  # Tick marks at the boundaries, which includes zero
        range=[-0.5, 1.5],
        dtick=0.3
    )
)
        # Create two columns to display charts side by side
        col1, col2 = st.columns([0.6,0.6])
        
        # Display the first chart (Index and Benchmark Levels)
        with col1:
            st.plotly_chart(fig_index_level)
        
        # Display the second chart (Cumulative Return)
        with col2:
            st.plotly_chart(fig_cumulative_return)
        

       

       # Now, display the remaining charts below the first two
        col3, col4 = st.columns([0.6, 0.6])

       # Display the remaining charts just below the first two (adjust these charts to fit)
        with col3:
           st.plotly_chart(fig_rnr)  # Replace this with your other chart (e.g., some other graph you want to display)

        with col4:
           st.plotly_chart(fig_ann_ret) 
#         # Add custom styling to annual_returns_df2
#         styled_annual_returns = annual_returns_df2.tail(6).style.applymap(
#         lambda x: 'background-color: lightblue;', subset=[index_name, benchmark_name]  # Adding light blue background to specific columns
#         ).highlight_max(axis=0, color='yellow')  # Highlight the maximum values in yellow

# # Add custom styling to rnr_benchmark_index2
#         styled_rnr_benchmark = rnr_benchmark_index2.head().style.applymap(
#         lambda x: 'background-color: lig;', subset=columns_to_process  # Apply a light green background to specific columns
#         ).highlight_min(axis=0, color='lightcoral')  # Highlight the minimum values in light red

# Create two columns to display tables side by side
        col5, col6 = st.columns([0.5, 0.5])  # Both columns will have equal width
        # Display the second table (rnr_benchmark_index) in the second column
        with col5:
            st.subheader('Risk-Return Profile')
            st.write(rnr_benchmark_index2)  # Display the first 5 rows of rnr_benchmark_index

# Display the first table (annual_returns_df) in the first column
        with col6:
            st.subheader('Annual Returns (Last 6 years)')
            st.dataframe(annual_returns_df2)  # Display the last 8 rows of annual_returns_df






# Run the Streamlit app
if __name__ == '__main__':
    main()
