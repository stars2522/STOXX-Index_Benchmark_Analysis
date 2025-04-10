# =========================================================
# LIBRARIES NEEDED
# =========================================================
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import function_v2 as functions
import os


# =========================================================
# FUNCTION TO CONVERT DATAFRAME INTO CSV
# =========================================================
# Function to convert DataFrame to CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False)

# =========================================================
# PAGE CONFIGURATION SETTINGS
# =========================================================
# Adjust the page configuration to have more space for charts
st.set_page_config(
    page_title="STOXX INDEX-BENCHMARK LEVEL ANALYSIS",
    page_icon="📊",
    layout="wide",  # "wide" layout increases the width of the content area
    initial_sidebar_state="expanded" # Expands the sidebar by default
 )

# ==================================
# CREATING THE SIDEBAR
# ==================================
st.sidebar.markdown(
    '<style> div[role="complementary"] { width: 200px; } </style>',
    unsafe_allow_html=True
)

# =========================================================
# DESIGNING THE STREAMLIT DASHBOARD OUTLINE
# =========================================================
# Streamlit app
def main():
    # Sidebar for user input
    st.sidebar.title('Index-Benchmark Level Analysis')
    
    index_file = st.sidebar.file_uploader("Upload the Index file (CSV or TXT)", type=["csv", "txt"])
    benchmark_file = st.sidebar.file_uploader("Upload the Benchmark file (CSV or TXT)", type=["csv", "txt"])
    
    # Automatically extract the index name from the file name (assuming the file name is in the format "h_{index_name}.txt")
    if index_file:
        index_name = os.path.splitext(index_file.name)[0]  # Remove file extension
        index_name = index_name.replace("h_", "").upper()  # Remove "h_" and convert to uppercase
        st.sidebar.write(f"Index Symbol: {index_name}")

    if benchmark_file:
    # Extract the benchmark name from the file name
        benchmark_name = os.path.splitext(benchmark_file.name)[0]  # Remove file extension
        benchmark_name = benchmark_name.replace("h_", "").upper()  # Remove "h_" and convert to uppercase
        st.sidebar.write(f"Benchmark Symbol: {benchmark_name}")

    # File uploader for Factor Attribution Report
    factor_file = st.sidebar.file_uploader("Upload the Factor Attribution report (Excel)", type=["xlsx", "xls"])
    # Main area
    st.title('Index-Benchmark Level Analysis')

# =========================================================
# CREATE INDEX AND BENCHMARK DATAFRAMES
# =========================================================
    # Proceed with processing if both files are uploaded 
    if index_file and benchmark_file and index_name and benchmark_name:
        # Process index and benchmark data
        index_data = functions.process_index_data(index_file, index_name)
        benchmark_data = functions.process_index_data(benchmark_file, benchmark_name)
        final_date=index_data.iloc[-1]['Date']
        # final_date = pd.to_datetime(final_date)
        final_date=final_date.strftime("%d %b %Y")


# ===============================================================================
# MERGE THE TWO TO GET COMMON DATES MASTER DATAFRAME, GIVES INDEX LEVEL DATAFRAME
# ===============================================================================
        # Merge both dataframes based on Date
        master_data = pd.merge(index_data, benchmark_data, on='Date', how='inner')
        # st.write(master_data.head(10))
        # List of columns to calculate returns and cumulative returns for
        names = [f'{index_name}', f'{benchmark_name}']
        # Loop through each name and calculate the returns and cumulative returns
        for name in names:
            ## REBASE INDEX LEVELS TO START FROM 100
            master_data[f'Indexvalue_{name}'] = (master_data[f'Indexvalue_{name}'] / master_data[f'Indexvalue_{name}'].iloc[0]) * 100
            ## RETURN CALCULATION
            master_data[f'Return_{name}'] = master_data[f'Indexvalue_{name}'].pct_change()
        
        ### Separate dataframes for Levels and Cumulative return data
        index_level_data=master_data[['Date',f'Indexvalue_{index_name}',f'Indexvalue_{benchmark_name}']]

# =======================================================================================
# DROPNA, CALCULATE CUMULATIVE RETURNS, NORMALISE THEM, GIVES CUMULATIVE RETURN DATAFRAME
# =======================================================================================
        master_data2=master_data.dropna()
        ## CUMULATIVE RETURN CALCULATION
        for name in names:
            master_data2[f'Cumulative_return_{name}'] = (1 + master_data2[f'Return_{name}']).cumprod() - 1
            ## NORMALISE CUMULATIVE RETURN TO START FROM 0
            master_data2[f'Cumulative_return_{name}'] -= master_data2[f'Cumulative_return_{name}'].iloc[0]
        
        cumulative_ret_data=master_data2[['Date',f'Cumulative_return_{index_name}',f'Cumulative_return_{benchmark_name}']]

# =======================================
# DATAFRAME FOR ANNUAL RETURNS (YEARLY)
# =======================================  
        ### ANNUAL RETURNS DATAFRAME CREATION
        annual_returns_df = functions.calculate_annual_returns(master_data, index_name, benchmark_name)
        annual_returns_df2=annual_returns_df[['Year',f'Annual_return_{index_name}',f'Annual_return_{benchmark_name}']]
        for name in names:
            ## REBASE INDEX LEVELS TO START FROM 100
            annual_returns_df2[f'Annual_return_{name}'] = annual_returns_df2[f'Annual_return_{name}'] * 100

# =========================================
# DATAFRAME FOR RNR (FOR DIFFERENT TIME PDS)
# =========================================  
        #### RNR DATAFRAME CREATION
        benchmark_data=master_data[['Date',f'Indexvalue_{benchmark_name}',f'Return_{benchmark_name}']]
        index_data=master_data[['Date',f'Indexvalue_{index_name}',f'Return_{index_name}']]
        
        period_dates_index = functions.get_period_dates(index_data)
        period_rnr_index = functions.get_rnr_values(index_data, period_dates_index, index_name)

        period_dates_benchmark = functions.get_period_dates(benchmark_data)
        period_rnr_benchmark = functions.get_rnr_values(benchmark_data, period_dates_benchmark, benchmark_name)

        # Merge RNR data for both index and benchmark
        rnr_benchmark_index = pd.merge(period_rnr_index, period_rnr_benchmark, 
                                       on=['Period', 'Start Date', 'End Date'])
        rnr_benchmark_index2=rnr_benchmark_index[['Period',f'Annualised Return_{index_name}',f'Annualised Return_{benchmark_name}',
                                                  f'Volatility_{index_name}',f'Volatility_{benchmark_name}']]
        for name in names:
            rnr_benchmark_index2[f'Annualised Return_{name}'] = rnr_benchmark_index2[f'Annualised Return_{name}'] * 100
            rnr_benchmark_index2[f'Volatility_{name}'] = rnr_benchmark_index2[f'Volatility_{name}'] * 100

# =====================
# CHART-1 INDEX LEVELS
# =====================   
        # print(px.colors.qualitative.Plotly)
        # PLOTTING
        fig_index_level = px.line(index_level_data, x='Date', y=[f'Indexvalue_{index_name}', f'Indexvalue_{benchmark_name}'],
                                  title='Index Levels',
                                  labels={'Date': 'Date', 
                                          f'{index_name}': f'Index Level: {index_name}', 
                                          f'{benchmark_name}': f'Index Level: {benchmark_name}'})
        # Update the legend names
        fig_index_level.update_traces(name=f"{index_name}", selector=dict(name=f'Indexvalue_{index_name}'))

        fig_index_level.update_traces(name=f"{benchmark_name}", selector=dict(name=f'Indexvalue_{benchmark_name}'))
        # Set the label for the y-axis
        fig_index_level.update_layout(legend_title=None,
            # height=1200,width=600,
                   yaxis_title="Index Level",  # You can adjust this label as per your need
                      )
        
        # Inspecting the color of the lines used in the figure
        for trace in fig_index_level['data']:
            print(f"Trace Name: {trace['name']}, Line Color: {trace['line']['color']}")

# ==========================
# CHART-2 CUMULATIVE RETURNS
# ==========================  
        # Plotting the Cumulative Return chart using Plotly
        fig_cumulative_return = px.line(cumulative_ret_data, x='Date', y=[f'Cumulative_return_{index_name}', f'Cumulative_return_{benchmark_name}'],
                                        title='Cumulative Returns',
                                        labels={'Date': 'Date', 
                                                f'Cumulative_return_{index_name}': f'Cumulative Return: {index_name}', 
                                                f'Cumulative_return_{benchmark_name}': f'Cumulative Return: {benchmark_name}'})
        fig_cumulative_return.update_layout(legend_title=None,
            # height=1200,width=600,
                   yaxis_title="Cumulative returns",yaxis=dict(tickformat='.0%'), # You can adjust this label as per your need
                      )
        # Update the legend names
        for name in names:
            fig_cumulative_return.update_traces(name=f"{name}", selector=dict(name=f'Cumulative_return_{name}'))

# =======================================
# CHART-3 RISK-RETURN METRICS (FACTSHEET)
# =======================================
        fig_rnr = go.Figure()

        # Add bar traces for the Annualized Returns of index and benchmark
        fig_rnr.add_trace(go.Bar(
            x=[i for i in range(len(rnr_benchmark_index))],
            y=rnr_benchmark_index[f'Annualised Return_{index_name}'],
            name=f'{index_name} Return',
            marker_color='#000001',
            width=0.4,marker=dict(line=dict(width=0))
        ))

        fig_rnr.add_trace(go.Bar(
            x=[i + 0.4 for i in range(len(rnr_benchmark_index))],
            y=rnr_benchmark_index[f'Annualised Return_{benchmark_name}'],
            name=f'{benchmark_name} Return',
            marker_color='#000002',
            width=0.4,marker=dict(line=dict(width=0))
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
              x=0.5,  # Position legend to the right of the plot (adjust as needed)
              y=-0.35,  # Position legend at the top
              xanchor='center',  # Center the legend horizontally
              yanchor='bottom',
              traceorder='normal',
              orientation='h',font=dict(size=10),
              itemwidth=50,  # Set width for each legend item (this can help prevent wrapping)
        tracegroupgap=8,
    ),margin=dict(t=80)
        )

# =======================================
# CHART-4 YEARLY ANNUAL RETURNS
# =======================================
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
        ticks='outside',  # Adds tick marks on the outside of the axis
        # tickson='boundaries',  # Tick marks at the boundaries, which includes zero
        range=[-0.5, 1.5],
        dtick=0.3
    ))
        
# =======================================
# CHARTS CONFIGURATION AND OUTLINE
# =======================================
# Start and End Date for the Index Levels Chart
        start_date_inl = index_level_data['Date'].iloc[0].strftime("%d %b %Y")
        end_date_inl = index_level_data['Date'].iloc[-1].strftime("%d %b %Y")
        # Create two columns to display charts side by side
        col1, col2 = st.columns([0.6,0.6])
        # Display the first chart (Index and Benchmark Levels)
        with col1:
            st.plotly_chart(fig_index_level)
            csv_index_level = convert_df_to_csv(index_level_data)
            st.markdown(f"<div style='text-align:center; margin-top: 10px; font-size: 12px;'>Date from {start_date_inl} till {end_date_inl}</div>", unsafe_allow_html=True)

            st.markdown("<div style='text-align:center; margin-top: 10px;'>", unsafe_allow_html=True)
            st.download_button(
            label="Download Index Levels Data",
            data=csv_index_level,
            file_name="Index_levels_data.csv",
            mime="text/csv"
        )
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Display the second chart (Cumulative Return)
        with col2:
            st.plotly_chart(fig_cumulative_return)
            csv_cumulative_ret = convert_df_to_csv(cumulative_ret_data)

            # Start and End Date for the Index Levels Chart
            start_date_cret = cumulative_ret_data['Date'].iloc[0].strftime("%d %b %Y")
            end_date_cret = cumulative_ret_data['Date'].iloc[-1].strftime("%d %b %Y")
            st.markdown(f"<div style='text-align:center; margin-top: 10px; font-size: 12px;'>Data from {start_date_cret} till {end_date_cret}</div>", unsafe_allow_html=True)

            st.markdown("<div style='text-align:center; margin-top: 10px;'>", unsafe_allow_html=True)
            st.download_button(
            label="Download Cumulative Returns Data",
            data=csv_cumulative_ret,
            file_name="Cumulative_returns_data.csv",
            mime="text/csv"
        )
            st.markdown("</div>", unsafe_allow_html=True)


       # Now, display the remaining charts below the first two
        col3, col4 = st.columns([0.6, 0.6])

        # Display the first chart (Risk-Return Profile)
        with col3:
            st.plotly_chart(fig_rnr)
            rnr_benchmark_index2.reset_index(inplace=True)
            csv_rnr = convert_df_to_csv(rnr_benchmark_index2)
            st.markdown(f"<div style='text-align:center; margin-top: 10px; font-size: 12px;'>Data as on {final_date} </div>", unsafe_allow_html=True)
        # Centered download button below the chart
            st.markdown("<div style='text-align:center; margin-top: 10px;'>", unsafe_allow_html=True)
            st.download_button(
            label="Download Risk-Return Data",
            data=csv_rnr,
            file_name="Risk_return_data.csv",
            mime="text/csv"
        )
            st.markdown("</div>", unsafe_allow_html=True)

    # Display the second chart (Annual Returns)
        with col4:
            st.plotly_chart(fig_ann_ret)
            annual_returns_df2.reset_index(inplace=True)
            csv_annual_ret = convert_df_to_csv(annual_returns_df2)
            st.markdown(f"<div style='text-align:center; margin-top: 10px; font-size: 12px;'>Data from {start_date_inl} till {end_date_inl}</div>", unsafe_allow_html=True)

        # Centered download button below the chart
            st.markdown("<div style='text-align:center; margin-top: 10px;'>", unsafe_allow_html=True)
            st.download_button(
            label="Download Annual Returns Data",
            data=csv_annual_ret,
            file_name="Annual_returns_data.csv",
            mime="text/csv"
        )
            st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# ADDITIONAL STEP: FACTOR FILE UPLOAD AFTER CHARTS
# ============================================================
        # =========================================================
        # FACTOR ATTRIBUTION ANALYSIS (IF FILE UPLOADED)
        # =========================================================
        if factor_file:
            # Load and process the factor attribution data
            fig_sector, fig_industry, fig_style, fig_country, fig_summary, fig_decomp,summary_df, decomp_df2, style_df, country_df, sector_df,industry_df3 = functions.factor_attribution_analysis(factor_file)
            
            st.title('Factor Attribution Analysis')

            # Create two columns for Summary and Decomposition charts
            col5, col6 = st.columns([0.6, 0.6])
            with col5:
                st.plotly_chart(fig_summary)
                csv_summary = convert_df_to_csv(summary_df)
                st.markdown("<div style='text-align:center; margin-top: 10px;'>", unsafe_allow_html=True)
                st.download_button(
                label="Download Active Risk-Return Summary",
                data=csv_summary,
                 file_name="summary_data.csv",
                 mime="text/csv"
                 )
                st.markdown("</div>", unsafe_allow_html=True)
            with col6:
                st.plotly_chart(fig_decomp)
                csv_decomp = convert_df_to_csv(decomp_df2)
                st.markdown("<div style='text-align:center; margin-top: 10px;'>", unsafe_allow_html=True)
                st.download_button(
                label="Download Active Return Decomposition Data",
                data=csv_decomp,
                 file_name="decomposition_data.csv",
                 mime="text/csv"
                 )
                st.markdown("</div>", unsafe_allow_html=True)
            
            # st.title('Contribution to Active Return')
            st.write("<h2 style='font-size: 20px;'>Contribution to Active Return (by different factors)</h2>", unsafe_allow_html=True)


            # Create another set of columns for Sector and Country charts
            col7, col8 = st.columns([0.6, 0.6])
            with col7:
                st.plotly_chart(fig_style)
                csv_style = convert_df_to_csv(style_df)
                st.markdown("<div style='text-align:center; margin-top: 10px;'>", unsafe_allow_html=True)
                st.download_button(
                label="Download Style Factor Contribution Data",
                data=csv_style,
                 file_name="style_factor_data.csv",
                 mime="text/csv"
                 )
                st.markdown("</div>", unsafe_allow_html=True)
            with col8:
                st.plotly_chart(fig_country)
                csv_country = convert_df_to_csv(country_df)
                st.markdown("<div style='text-align:center; margin-top: 10px;'>", unsafe_allow_html=True)
                st.download_button(
                label="Download Country Factor Contribution Data",
                data=csv_country,
                 file_name="country_factor_data.csv",
                 mime="text/csv"
                 )
                st.markdown("</div>", unsafe_allow_html=True)

            # Finally, another set of columns for the Sector and Industry charts
            col9, col10 = st.columns([0.6, 0.6])
            with col9:
                st.plotly_chart(fig_sector)
                csv_sector = convert_df_to_csv(sector_df)
                st.markdown("<div style='text-align:center; margin-top: 10px;'>", unsafe_allow_html=True)
                st.download_button(
                label="Download Sector Factor Contribution Data",
                data=csv_sector,
                 file_name="sector_factor_data.csv",
                 mime="text/csv"
                 )
                st.markdown("</div>", unsafe_allow_html=True)
            with col10:
                st.plotly_chart(fig_industry)
                csv_industry = convert_df_to_csv(industry_df3)
                st.markdown("<div style='text-align:center; margin-top: 10px;'>", unsafe_allow_html=True)
                st.download_button(
                label="Download Industry Factor Contribution Data",
                data=csv_industry,
                 file_name="industry_factor_data.csv",
                 mime="text/csv"
                 )
                st.markdown("</div>", unsafe_allow_html=True)
# Run the Streamlit app
if __name__ == '__main__':
    main()
