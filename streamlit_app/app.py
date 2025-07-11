import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="🍽️ Restaurant Visitor Forecast",
    layout="wide",
    page_icon="🍴",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS FOR MODERN UI
# =========================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #667eea;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-left: 4px solid #28a745;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .forecast-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODELS AND DATA
# =========================
@st.cache_data
def load_encoder():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(app_dir, 'models', 'store_label_encoder.pkl')
    return joblib.load(model_path)

@st.cache_data
def load_model():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(app_dir, 'models', 'xgb_visitor_model.pkl')
    return joblib.load(model_path)

@st.cache_data
def load_store_data():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_dir, 'data', 'air_store_info.csv')
    return pd.read_csv(data_path)

@st.cache_data
def load_visit_data():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_dir, 'data', 'air_visit_data.csv')
    return pd.read_csv(data_path)

@st.cache_data
def load_store_stats():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_dir, 'data', 'store_stats.csv')
    return pd.read_csv(data_path)

# =========================
# ENHANCED DATA LOADING WITH ERROR HANDLING
# =========================
@st.cache_data
def load_all_data():
    try:
        with st.spinner("Loading data and models..."):
            encoder = load_encoder()
            model = load_model()
            store_df = load_store_data()
            stats_df = load_store_stats()
            visit_df = load_visit_data()
            
            # Data validation
            if any(df.empty for df in [store_df, stats_df, visit_df]):
                st.error("Some data files are empty. Please check your data.")
                return None
            
            return encoder, model, store_df, stats_df, visit_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please check if all model and data files are properly uploaded.")
        return None

# Load data
data = load_all_data()
if data is None:
    st.stop()

encoder, model, store_df, stats_df, visit_df = data

# =========================
# MODERN HEADER WITH ANIMATIONS
# =========================
st.markdown("""
<div class="main-header">
    <h1>🍽️ Restaurant Visitor Forecasting Dashboard</h1>
    <p>Predict restaurant traffic with AI-powered forecasting for smarter business decisions</p>
</div>
""", unsafe_allow_html=True)

# =========================
# DASHBOARD METRICS
# =========================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Restaurants",
        value=f"{len(store_df):,}",
        delta=f"{len(store_df['air_area_name'].unique())} Areas"
    )

with col2:
    st.metric(
        label="Genres Available",
        value=f"{len(store_df['air_genre_name'].unique())}",
        delta="All Categories"
    )

with col3:
    total_visits = visit_df['visitors'].sum()
    st.metric(
        label="Total Historical Visits",
        value=f"{total_visits:,}",
        delta=f"Avg: {visit_df['visitors'].mean():.0f}/day"
    )

with col4:
    date_range = pd.to_datetime(visit_df['visit_date']).max() - pd.to_datetime(visit_df['visit_date']).min()
    st.metric(
        label="Data Period",
        value=f"{date_range.days} days",
        delta="Historical Data"
    )

# =========================
# ENHANCED SIDEBAR WITH BETTER UX
# =========================
st.sidebar.markdown("Forecasting Controls")

# Store selection with search
selected_genre = st.sidebar.selectbox(
    "Filter by Genre", 
    ['All'] + sorted(store_df['air_genre_name'].unique().tolist()),
    help="Choose a restaurant genre to filter options"
)

selected_area = st.sidebar.selectbox(
    "Filter by Area", 
    ['All'] + sorted(store_df['air_area_name'].unique().tolist()),
    help="Select a specific area to narrow down restaurants"
)

# Apply filters
filtered_df = store_df.copy()
if selected_genre != 'All':
    filtered_df = filtered_df[filtered_df['air_genre_name'] == selected_genre]
if selected_area != 'All':
    filtered_df = filtered_df[filtered_df['air_area_name'] == selected_area]

# Enhanced store selection
store_options = []
for _, row in filtered_df.iterrows():
    store_options.append(f"{row['air_store_id']} - {row['air_genre_name']} ({row['air_area_name']})")

if not store_options:
    st.sidebar.error("No restaurants match your filters!")
    st.stop()

selected_store_display = st.sidebar.selectbox(
    "Select Restaurant",
    store_options,
    help="Choose a restaurant for detailed analysis"
)

# Extract store ID from display string
selected_store_id = selected_store_display.split(' - ')[0]

# Multi-store comparison option
st.sidebar.markdown("---")
enable_comparison = st.sidebar.checkbox(
    "Enable Multi-Store Comparison",
    help="Compare multiple restaurants side by side"
)

comparison_stores = []
if enable_comparison:
    comparison_stores = st.sidebar.multiselect(
        "Select stores to compare",
        [store.split(' - ')[0] for store in store_options],
        default=[selected_store_id]
    )

# =========================
# ENHANCED DATE SELECTION
# =========================
st.sidebar.markdown("Forecast Period")

# Quick date options
date_option = st.sidebar.radio(
    "Choose forecast period:",
    ["Next 7 days", "Next 30 days", "Custom range"]
)

if date_option == "Next 7 days":
    start_date = datetime.today().date()
    end_date = start_date + timedelta(days=6)
elif date_option == "Next 30 days":
    start_date = datetime.today().date()
    end_date = start_date + timedelta(days=29)
else:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.today().date())
    with col2:
        end_date = st.date_input("End Date", value=datetime.today().date() + timedelta(days=6))

# Date validation
if end_date < start_date:
    st.sidebar.error("End date cannot be before start date!")
    st.stop()

if (end_date - start_date).days > 365:
    st.sidebar.warning("Forecast period is very long. Consider shorter periods for better accuracy.")

# =========================
# ENHANCED FEATURE PREPARATION
# =========================
def prepare_features(store_id, dates):
    """Enhanced feature preparation with error handling"""
    try:
        df = pd.DataFrame({
            'air_store_id': [store_id] * len(dates),
            'visit_date': dates
        })
        
        # Date features
        df['day_of_week'] = df['visit_date'].dt.dayofweek
        df['year'] = df['visit_date'].dt.year
        df['month'] = df['visit_date'].dt.month
        df['day'] = df['visit_date'].dt.day
        df['week'] = df['visit_date'].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Merge with store stats
        df = df.merge(stats_df, on='air_store_id', how='left')
        
        # Handle missing store stats
        if df['mean_visitors'].isna().any():
            avg_mean = stats_df['mean_visitors'].mean()
            df['mean_visitors'].fillna(avg_mean, inplace=True)
            df['median_visitors'].fillna(stats_df['median_visitors'].mean(), inplace=True)
            df['min_visitors'].fillna(stats_df['min_visitors'].mean(), inplace=True)
            df['max_visitors'].fillna(stats_df['max_visitors'].mean(), inplace=True)
            df['std_visitors'].fillna(stats_df['std_visitors'].mean(), inplace=True)
        
        # Encode store ID
        df['store_id_encoded'] = encoder.transform(df['air_store_id'])
        
        return df
    except Exception as e:
        st.error(f"Error preparing features: {str(e)}")
        return None

# =========================
# STORE INFORMATION DISPLAY
# =========================
if selected_store_id in store_df['air_store_id'].values:
    store_info = store_df[store_df['air_store_id'] == selected_store_id].iloc[0]
    
    st.markdown("Selected Restaurant Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="info-box">
            <h4>Location Details</h4>
            <p><strong>Area:</strong> {store_info['air_area_name']}</p>
            <p><strong>Store ID:</strong> {store_info['air_store_id']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
            <h4>Restaurant Type</h4>
            <p><strong>Genre:</strong> {store_info['air_genre_name']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Calculate store-specific metrics
        if selected_store_id in stats_df['air_store_id'].values:
            store_stats = stats_df[stats_df['air_store_id'] == selected_store_id].iloc[0]
            st.markdown(f"""
            <div class="info-box">
                <h4>Performance Metrics</h4>
                <p><strong>Avg Visitors:</strong> {store_stats['mean_visitors']:.0f}</p>
                <p><strong>Max Visitors:</strong> {store_stats['max_visitors']:.0f}</p>
            </div>
            """, unsafe_allow_html=True)

# =========================
# MAIN FORECASTING SECTION
# =========================
st.markdown("---")
st.markdown("AI-Powered Visitor Forecast")

# Create forecast button with enhanced styling
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Generate Forecast", key="main_forecast", help="Click to generate visitor predictions"):
        
        # Input validation
        if not selected_store_id:
            st.error("Please select a restaurant first!")
        else:
            try:
                with st.spinner("AI is analyzing patterns and generating forecasts..."):
                    # Generate date range
                    date_range = pd.date_range(start=start_date, end=end_date)
                    
                    # Prepare features
                    forecast_df = prepare_features(selected_store_id, date_range)
                    
                    if forecast_df is not None:
                        # Make predictions
                        features = forecast_df[[
                            'year', 'month', 'day', 'day_of_week', 'week', 'is_weekend',
                            'mean_visitors', 'median_visitors', 'min_visitors', 'max_visitors',
                            'std_visitors', 'store_id_encoded'
                        ]]
                        
                        predictions = model.predict(features)
                        forecast_df['Predicted Visitors'] = predictions.round().astype(int)
                        
                        # Display results with enhanced styling
                        st.markdown("""
                        <div class="success-box">
                            <h4>Forecast Generated Successfully!</h4>
                            <p>Your AI-powered visitor predictions are ready.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Summary metrics
                        total_predicted = forecast_df['Predicted Visitors'].sum()
                        avg_predicted = forecast_df['Predicted Visitors'].mean()
                        max_day = forecast_df.loc[forecast_df['Predicted Visitors'].idxmax(), 'visit_date'].strftime('%A, %B %d')
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Predicted Visitors", f"{total_predicted:,}")
                        with col2:
                            st.metric("Daily Average", f"{avg_predicted:.0f}")
                        with col3:
                            st.metric("Busiest Day", max_day)
                        
                        # Interactive Plotly chart
                        st.markdown("Interactive Forecast Chart")
                        
                        fig = go.Figure()
                        
                        # Add forecast line
                        fig.add_trace(go.Scatter(
                            x=forecast_df['visit_date'],
                            y=forecast_df['Predicted Visitors'],
                            mode='lines+markers',
                            name='Predicted Visitors',
                            line=dict(color='#667eea', width=3),
                            marker=dict(size=8, color='#667eea'),
                            hovertemplate='<b>Date:</b> %{x}<br><b>Predicted Visitors:</b> %{y}<extra></extra>'
                        ))
                        
                        # Add weekend highlighting
                        weekend_dates = forecast_df[forecast_df['is_weekend'] == 1]['visit_date']
                        weekend_visitors = forecast_df[forecast_df['is_weekend'] == 1]['Predicted Visitors']
                        
                        fig.add_trace(go.Scatter(
                            x=weekend_dates,
                            y=weekend_visitors,
                            mode='markers',
                            name='Weekend Days',
                            marker=dict(size=12, color='#ff6b6b', symbol='diamond'),
                            hovertemplate='<b>Weekend:</b> %{x}<br><b>Predicted Visitors:</b> %{y}<extra></extra>'
                        ))
                        
                        fig.update_layout(
                            title='Visitor Forecast with Weekend Highlights',
                            xaxis_title='Date',
                            yaxis_title='Predicted Visitors',
                            hovermode='x unified',
                            showlegend=True,
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed forecast table
                        st.markdown("Detailed Forecast Table")
                        
                        display_df = forecast_df.copy()
                        display_df['visit_date'] = display_df['visit_date'].dt.strftime('%Y-%m-%d')
                        display_df['Day of Week'] = pd.to_datetime(display_df['visit_date']).dt.day_name()
                        display_df['Is Weekend'] = display_df['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
                        
                        table_df = display_df[['visit_date', 'Day of Week', 'Is Weekend', 'Predicted Visitors']].rename(columns={
                            'visit_date': 'Date',
                            'Predicted Visitors': 'Predicted Visitors'
                        })
                        
                        st.dataframe(table_df, use_container_width=True)
                        
                        # Download option
                        csv = table_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Forecast as CSV",
                            data=csv,
                            file_name=f"forecast_{selected_store_id}_{start_date}_to_{end_date}.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Forecast generation failed: {str(e)}")
                st.info("Please try again or contact support if the issue persists.")

# =========================
# ENHANCED HISTORICAL ANALYSIS
# =========================
st.markdown("---")
st.markdown("Historical Performance Analysis")

try:
    hist_visits = visit_df[visit_df['air_store_id'] == selected_store_id].copy()
    
    if not hist_visits.empty:
        hist_visits['visit_date'] = pd.to_datetime(hist_visits['visit_date'])
        hist_visits = hist_visits.sort_values('visit_date')
        
        # Historical metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Days", len(hist_visits))
        with col2:
            st.metric("Avg Daily Visitors", f"{hist_visits['visitors'].mean():.0f}")
        with col3:
            st.metric("Peak Day", f"{hist_visits['visitors'].max()}")
        with col4:
            st.metric("Lowest Day", f"{hist_visits['visitors'].min()}")
        
        # Interactive historical chart
        st.markdown("Historical Visitor Trends")
        
        fig = px.line(
            hist_visits, 
            x='visit_date', 
            y='visitors',
            title='Historical Visitor Patterns',
            labels={'visit_date': 'Date', 'visitors': 'Visitors'}
        )
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Visitors',
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ENHANCED HEATMAP SECTION
        st.markdown("Enhanced Visitor Pattern Heatmap")
        
        # Prepare heatmap data
        hist_visits['day_of_week'] = hist_visits['visit_date'].dt.day_name()
        hist_visits['month'] = hist_visits['visit_date'].dt.strftime('%b')
        hist_visits['hour'] = hist_visits['visit_date'].dt.hour  # If you have hourly data
        
        # Create pivot table for heatmap
        pivot_table = hist_visits.pivot_table(
            index='day_of_week', 
            columns='month', 
            values='visitors', 
            aggfunc='mean'
        )
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_table = pivot_table.reindex(day_order)
        
        # Create interactive heatmap with Plotly
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='YlOrRd',
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br><b>%{x}</b><br>Avg Visitors: %{z:.0f}<extra></extra>'
        ))
        
        fig_heatmap.update_layout(
            title='Average Visitors by Day of Week and Month',
            xaxis_title='Month',
            yaxis_title='Day of Week',
            height=500
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Insights from heatmap
        busiest_day = pivot_table.mean(axis=1).idxmax()
        busiest_month = pivot_table.mean(axis=0).idxmax()
        
        st.markdown(f"""
        <div class="info-box">
            <h4>Pattern Insights</h4>
            <p><strong>Busiest Day:</strong> {busiest_day}</p>
            <p><strong>Peak Month:</strong> {busiest_month}</p>
            <p><strong>Weekend vs Weekday:</strong> 
            {('Weekends are busier' if pivot_table.loc[['Saturday', 'Sunday']].mean().mean() > 
              pivot_table.loc[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].mean().mean() 
              else 'Weekdays are busier')}</p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.info("No historical data available for this restaurant.")
        
except Exception as e:
    st.warning(f"Error loading historical data: {str(e)}")

# =========================
# MULTI-STORE COMPARISON (if enabled)
# =========================
if enable_comparison and len(comparison_stores) > 1:
    st.markdown("---")
    st.markdown("Multi-Store Performance Comparison")
    
    try:
        comparison_data = []
        
        for store_id in comparison_stores:
            store_visits = visit_df[visit_df['air_store_id'] == store_id]
            store_name = store_df[store_df['air_store_id'] == store_id]['air_genre_name'].iloc[0]
            
            comparison_data.append({
                'Store ID': store_id,
                'Genre': store_name,
                'Avg Visitors': store_visits['visitors'].mean(),
                'Max Visitors': store_visits['visitors'].max(),
                'Total Days': len(store_visits)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Comparison chart
        fig_comparison = px.bar(
            comparison_df,
            x='Store ID',
            y='Avg Visitors',
            title='Average Visitors Comparison',
            color='Genre',
            hover_data=['Max Visitors', 'Total Days']
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Comparison table
        st.dataframe(comparison_df, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Error in comparison analysis: {str(e)}")

# =========================
# FOOTER AND ADDITIONAL INFO
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Powered by AI Machine Learning | Built with Streamlit</p>
    <p><strong>Pro Tip:</strong> Use the heatmap to identify peak days and optimize your staffing!</p>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR FOOTER
# =========================
st.sidebar.markdown("---")
st.sidebar.markdown("Model Performance")
st.sidebar.info("This AI model is trained on historical visitor data to provide accurate predictions.")

st.sidebar.markdown("Usage Tips")
st.sidebar.markdown("""
- Use short forecast periods (7-30 days) for best accuracy
- Check the heatmap for seasonal patterns
- Compare multiple stores for insights
- Download forecasts for business planning
""")

st.sidebar.markdown("Need Help?")
st.sidebar.markdown("Contact support for questions about forecasting or data interpretation.")