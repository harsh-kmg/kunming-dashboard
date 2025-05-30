import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime
from io import BytesIO
import streamlit as st
from utility import *
def load_data(file):
    file_type = file.split('.')[-1]
    if file_type == 'csv':
        return pd.read_csv(file)
    elif file_type == 'xlsx':
        df=pd.read_excel(file,sheet_name=None)
        df_dict = {}
        for sheet_name, df_ in df.items():
            df_dict[sheet_name] = df_
        return df_dict
    
    elif file_type == 'db':
        from sqlalchemy import create_engine
        cnx = create_engine('sqlite:///kunmings.db').connect()
        file_name = file.split('.')[0]
        df = pd.read_sql(file_name, cnx)
        cnx.close()
        return df
def save_data(df):
    from sqlalchemy import create_engine
    cnx = create_engine('sqlite:///kunmings.db').connect()
    df.to_sql('kunmings', cnx, index=False, if_exists='replace')
    cnx.close()
def create_bucket(df,stock_bucket=stock_bucket):
    """
    df : Monthly Stock Data Sheet
    stock_bucket : Dictionary containing bucket ranges
    """
    for key , values in stock_bucket.items():
        lower_bound , upper_bound = values
        index = df[(df['Weight']>=lower_bound) & (df['Weight']<upper_bound)].index.tolist()
        df.loc[index,'Buckets'] = key
    return df

def calculate_avg(df):
    """
    df : Monthly Stock Data Sheet
    """
    df['avg'] = df['Weight'] * df['Average\nCost\n(USD)']
    return df

def create_date_join(df):
    """
    df : Monthly Stock Data Sheet
    """
    df['Month'] = pd.to_datetime('today').month_name()
    df['Year'] = pd.to_datetime('today').year
    df['Join'] = df['Month'].astype(str) + '-' + df['Year'].map(lambda x: x-2000).astype(str)
    return df
def concatenate_first_two_rows(df):
    result = {}
    for col in df.columns:
        value1 = str(df.iloc[0][col])
        value2 = str(df.iloc[1][col])
        result[col] = f"{value1}_{value2}"
    return result
def populate_max_qty(df,MONTHLY_STOCK_DATA):
    """
    df : Max Qty Sheet
    MONTHLY_STOCK_DATA : Monthly Stock Data Sheet
    """
    columns=list(concatenate_first_two_rows(df.iloc[0:2,2:]).values())
    columns = ['Months','Buckets'] + columns
    df.columns = columns
    df=df.iloc[2:,:]
    df.reset_index(drop=True,inplace=True)
    _MAX_QTY_ = []
    MONTHLY_STOCK_DATA['Max Qty'] = None
    for indx, row in MONTHLY_STOCK_DATA.iterrows():
        join = row['Join']
        Shape = row['Shape key']
        Color = row['Color Key']
        Bucket = row['Buckets']
        if pd.isna(Color):
            value = None
        else:
            col_name = f"{Shape}_{Color}"
            value = df[(df['Months'] == join) & (df['Buckets'] == Bucket)][col_name].values.tolist()
        _MAX_QTY_.append(value)
    MONTHLY_STOCK_DATA['Max Qty'] = _MAX_QTY_
    MONTHLY_STOCK_DATA['Max Qty']=MONTHLY_STOCK_DATA['Max Qty'].map(lambda x:x[0] if isinstance(x, list) and len(x) > 0 else 0)
    return MONTHLY_STOCK_DATA

def populate_min_qty(df,MONTHLY_STOCK_DATA):
    """
    df : Buying Min Qty Sheet
    MONTHLY_STOCK_DATA : Monthly Stock Data Sheet
    """
    columns=list(concatenate_first_two_rows(df.iloc[0:2,2:]).values())
    columns = ['Months','Buckets'] + columns
    df.columns = columns
    df=df.iloc[2:,:]
    df.reset_index(drop=True,inplace=True)
    _MIN_QTY_ = []
    MONTHLY_STOCK_DATA['Min Qty'] = None
    for _, row in MONTHLY_STOCK_DATA.iterrows():
        join = row['Join']
        Shape = row['Shape key']
        Color = row['Color Key']
        Bucket = row['Buckets']
        if pd.isna(Color):
            value = None
        else:
            col_name = f"{Shape}_{Color}"
            value = df[(df['Months'] == join) & (df['Buckets'] == Bucket)][col_name].values.tolist()
        _MIN_QTY_.append(value)
    MONTHLY_STOCK_DATA['Min Qty'] = _MIN_QTY_
    MONTHLY_STOCK_DATA['Min Qty']=MONTHLY_STOCK_DATA['Min Qty'].map(lambda x:x[0] if isinstance(x, list) and len(x) > 0 else 0)
    return MONTHLY_STOCK_DATA

def populate_buying_prices(df,MONTHLY_STOCK_DATA):
    """
    df : Buying Max Prices Sheet 
    MONTHLY_STOCK_DATA : Monthly Stock Data Sheet
    """
    columns=list(concatenate_first_two_rows(df.iloc[0:2,2:]).values())
    columns = ['Months','Buckets'] + columns
    df.columns = columns
    df=df.iloc[2:,:]
    df.reset_index(drop=True,inplace=True)
    _BUYING_PRICE_ = []
    MONTHLY_STOCK_DATA['Max Buying Price'] = None
    for indx, row in MONTHLY_STOCK_DATA.iterrows():
        join = row['Join']
        Shape = row['Shape key']
        Color = row['Color Key']
        Bucket = row['Buckets']
        if pd.isna(Color):
            value = None
        else:
            col_name = f"{Shape}_{Color}"
            value = df[(df['Months'] == join) & (df['Buckets'] == Bucket)][col_name].values.tolist()
        _BUYING_PRICE_.append(value)
    MONTHLY_STOCK_DATA['Max Buying Price'] = _BUYING_PRICE_
    MONTHLY_STOCK_DATA['Max Buying Price']=MONTHLY_STOCK_DATA['Max Buying Price'].map(lambda x:x[0] if isinstance(x, list) and len(x) > 0 else 0)
    return MONTHLY_STOCK_DATA
def calculate_buying_price_avg(df):
    df['Buying Price Avg'] = df['Max Buying Price'] * df['Weight']
    return df

def get_quarter(month):
    Quarter_Month_Map = {
    'Q1': ['January', 'February', 'March'],
    'Q2': ['April', 'May', 'June'],
    'Q3': ['July', 'August', 'September'],
    'Q4': ['October', 'November', 'December']
    }
    year = pd.to_datetime('today').year
    yr = year - 2000

    if month in Quarter_Month_Map['Q1']:
        return f'Q1-{yr}'
    elif month in Quarter_Month_Map['Q2']:
        return f'Q2-{yr}'
    elif month in Quarter_Month_Map['Q3']:
        return f'Q3-{yr}'
    elif month in Quarter_Month_Map['Q4']:
        return f'Q4-{yr}'
    else:
        return None
def populate_quarter(df):
    """
    df : Monthly Stock Data Sheet
    """
    df['Quarter'] = df['Month'].apply(get_quarter)
    return df

def poplutate_monthly_stock_sheet(file):
    """
    df_stock : Monthly Stock Data Sheet
    df_buying : Buying Max Prices Sheet
    df_min_qty : Buying Min Qty Sheet
    df_max_qty : Max Qty Sheet
    """
    df = load_data(file)
    df_stock = df['Monthly Stock Data']
    df_buying = df['Buying Max Prices']
    df_min_qty = df['MIN Data']
    df_max_qty = df['MAX Data']
    if df_stock.empty or df_buying.empty or df_min_qty.empty or df_max_qty.empty:
        raise ValueError("One or more dataframes are empty. Please check the input files.")
    df_stock = create_date_join(df_stock)
    df_stock = populate_quarter(df_stock)
    df_stock = calculate_avg(df_stock)
    df_stock = create_bucket(df_stock)
    df_stock = populate_max_qty(df_max_qty, df_stock)
    df_stock = populate_min_qty(df_min_qty, df_stock)
    df_stock = populate_buying_prices(df_buying, df_stock)
    df_stock = calculate_buying_price_avg(df_stock)
    return df_stock
def calculate_qoq_variance_percentage(current_quarter_price, previous_quarter_price):
    """
    Calculate quarter-on-quarter variance percentage of price.
    
    Args:
        current_quarter_price (float): Price for the current quarter
        previous_quarter_price (float): Price for the previous quarter
    
    Returns:
        float: Variance percentage (positive for increase, negative for decrease)
        
    Raises:
        ValueError: If previous quarter price is zero or negative
        TypeError: If inputs are not numeric
    """
    # Input validation
    if not isinstance(current_quarter_price, (int, float)) or not isinstance(previous_quarter_price, (int, float)):
        raise TypeError("Both prices must be numeric values")
    
    if previous_quarter_price <= 0:
        variance_percentage = 0.00001
        # raise ValueError("Previous quarter price must be positive (cannot be zero or negative)")
    
    # Calculate variance percentage
    if previous_quarter_price !=0:
        variance_percentage = ((current_quarter_price - previous_quarter_price) / previous_quarter_price) * 100
    else:
        variance_percentage = ((current_quarter_price - previous_quarter_price) / (previous_quarter_price+current_quarter_price)) * 100
    return round(variance_percentage, 2)


def calculate_qoq_variance_series(price_data):
    """
    Calculate quarter-on-quarter variance for a series of quarterly prices.
    
    Args:
        price_data (list): List of quarterly prices in chronological order
    
    Returns:
        list: List of QoQ variance percentages (starts from Q2 since Q1 has no previous quarter)
    """
    if len(price_data) < 2:
        raise ValueError("Need at least 2 quarters of data to calculate variance")
    
    variances = []
    for i in range(1, len(price_data)):
        variance = calculate_qoq_variance_percentage(price_data[i], price_data[i-1])
        variances.append(variance)
    
    return variances
def monthly_variance(df,col):
    analysis=df.groupby(['Month','Year'],as_index=False)[col].sum()
    analysis['Num_Month'] = analysis['Month'].map(month_map)
    analysis.sort_values(by=['Year','Num_Month'],inplace=True)
    analysis['Monthly_change']=analysis[col].pct_change().fillna(0).round(2)*100
    analysis['qaurter_change']=[0]+calculate_qoq_variance_series(analysis[col].tolist())
    return analysis


def get_filtered_data(FILTER_MONTH,FILTE_YEAR,FILTER_SHAPE,FILTER_COLOR,FILTER_BUCKET,FILTER_MONTHLY_VAR_COL):
    """
    file : Monthly Stock Data Sheet
    FILTER_MONTH : Month to filter
    FILTE_YEAR : Year to filter
    FILTER_SHAPE : Shape Key to filter
    FILTER_COLOR : Color Key to filter
    FILTER_BUCKET : Buckets to filter
    FILTER_MONTHLY_VAR_COL : Column to calculate monthly variance
    PARENT_DF : Parent DataFrame to concatenate with the monthly stock data
    """
    master_df = load_data('kunmings.db')
    filter_data=master_df[(master_df['Month'] == FILTER_MONTH) & \
                                      (master_df['Year'] == FILTE_YEAR) & \
                                        (master_df['Shape key'] == FILTER_SHAPE) &\
                                        (master_df['Color Key'] == FILTER_COLOR) &\
                                        (master_df['Buckets'] == FILTER_BUCKET)]
    max_buying_price = filter_data['Max Buying Price'].max()
    current_avg_cost = sum(.9*((filter_data['Max Buying Price'] * filter_data['Weight'])/(filter_data['Weight'].sum() if filter_data['Weight'].sum() != 0 else 1)))
    avg_value = master_df[FILTER_MONTHLY_VAR_COL].mean()
    MOM_Variance = (sum((filter_data[FILTER_MONTHLY_VAR_COL] - avg_value)/ avg_value )/filter_data.shape[0]) * 100
    var_analysis = monthly_variance(master_df,FILTER_MONTHLY_VAR_COL)
    MOM_Percent_Change = var_analysis[(var_analysis['Month'] == FILTER_MONTH) & (var_analysis['Year'] == FILTE_YEAR)]['Monthly_change'].values.tolist()[0]
    MOM_QoQ_Percent_Change = var_analysis[(var_analysis['Month'] == FILTER_MONTH) & (var_analysis['Year'] == FILTE_YEAR)]['qaurter_change'].values.tolist()[0]
    if MOM_Percent_Change == np.inf:
        MOM_Percent_Change = 0
    if MOM_QoQ_Percent_Change == np.inf:
        MOM_QoQ_Percent_Change = 0
    return [filter_data,int(max_buying_price),int(current_avg_cost), int(MOM_Variance), MOM_Percent_Change, MOM_QoQ_Percent_Change]

def get_final_data(file,PARENT_DF = 'kunmings.db'):
    df = poplutate_monthly_stock_sheet(file)
    parent_df = load_data(PARENT_DF)
    master_df = pd.concat([df, parent_df], ignore_index=True,axis=0)
    save_data(master_df)
    return master_df

def main():
    st.set_page_config(page_title="Yellow Diamond Dashboard", layout="wide")
    st.title("Yellow Diamond Dashboard")
    st.markdown("Upload Excel files to process multiple sheets and filter data.")
    # Initialize session state
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'master_df' not in st.session_state:
        st.session_state.master_df = pd.DataFrame()
    # Sidebar for controls
    st.sidebar.header("Controls")
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel File",
        type=['xlsx', 'xls'],
        help="Upload an Excel file with multiple sheets"
    )
    # Main content area
    if uploaded_file is not None and not st.session_state.data_processed:
        with st.spinner("Processing Excel file..."):
            st.subheader("🗄️ Master Database")
            st.session_state.master_df  = get_final_data(uploaded_file.name)
            st.session_state.data_processed = True
    if not st.session_state.master_df.empty or uploaded_file is not None:
        Month,Year,Shape,Color,Bucket,Variance_Column = st.columns(6)
        with Month:
            categories = ["None"]+list(st.session_state.master_df['Month'].unique())
            selected_month = st.selectbox("Filter by Month", categories)
        with Year:
            years = ["None"]+list(st.session_state.master_df['Year'].unique())
            selected_year = st.selectbox("Filter by Year", years)
        with Shape:
            shapes = ["None"]+list(st.session_state.master_df['Shape key'].unique())
            selected_shape = st.selectbox("Filter by Shape", shapes)
        with Color:
            colors = ["None"]+list(st.session_state.master_df['Color Key'].unique())
            selected_color = st.selectbox("Filter by Color", colors)
        with Bucket:
            buckets = ["None"]+list(st.session_state.master_df['Buckets'].unique())
            selected_bucket = st.selectbox("Filter by Bucket", buckets)
        with Variance_Column:
            variance_columns = ["None"]+['Buying Price Avg','Max Buying Price']
            selected_variance_column = st.selectbox("Select Variance Column", variance_columns)
        # Apply filters
        filtered_df = st.session_state.master_df.copy()
        if (selected_month != "None") & (selected_year != "None") & (selected_shape != "None") & (selected_color != "None") & (selected_bucket != "None") & (selected_variance_column != "None"):
            filter_data,max_buying_price,current_avg_cost,MOM_Variance,MOM_Percent_Change,MOM_QoQ_Percent_Change = get_filtered_data(selected_month,\
                                                                                                                        int(selected_year),\
                                                                                                                        selected_shape,\
                                                                                                                        selected_color,\
                                                                                                                        selected_bucket,\
                                                                                                                        selected_variance_column)
            # Display summary metrics
            st.subheader("📊 Summary Metrics")
            mbp,cac,mom_var,mom_perc,qoq_perc = st.columns(5)
            with mbp:
                st.metric("Max Buying Price", f"${max_buying_price:,.2f}")
            with cac:
                st.metric("Current Avg Cost", f"${current_avg_cost:,.2f}")
            with mom_var:
                st.metric("MOM Variance ", f"{MOM_Variance:,.2f}%")
            with mom_perc:
                st.metric("MOM Percent Change", f"{MOM_Percent_Change:.2f}%")
            with qoq_perc:
                st.metric("MOM QoQ Percent Change", f"{MOM_QoQ_Percent_Change:.2f}%")
            st.subheader("📊 Data Table")
            st.dataframe(
                filter_data,
                use_container_width=True,
                hide_index=True
                    )
            # Download processed data
            st.subheader("💾 Download Filtered Data")
            csv = filter_data.loc[:,['Product Id','Shape key','Color Key','avg','Min Qty','Max Qty','Buying Price Avg','Max Buying Price']].to_csv(index=False)
            st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
            )
            st.subheader("💾 Download Master Data")
            csv = filtered_df.to_csv(index=False)
            st.download_button(
            label="Download Master Data as CSV",
            data=csv,
            file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
            )
        else:
            
            st.info("No data in master database. Upload an Excel file to get started!")
    else:
        st.info("No data in master database. Upload an Excel file to get started!")
    # Reset button
    if st.sidebar.button("Reset Data Processing"):
        st.session_state.data_processed = False
        st.session_state.master_df = pd.DataFrame()
        st.rerun()
    
if __name__ == "__main__":
    main()