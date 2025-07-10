import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="🍽️ Restaurant Visitor Forecast",
    layout="wide",
    page_icon="🍴"
)

# =========================
# LOAD MODELS AND DATA
# =========================
@st.cache_resource
def load_model():
    return joblib.load('../models/xgb_visitor_model.pkl')

@st.cache_resource
def load_encoder():
    return joblib.load('../models/store_label_encoder.pkl')

@st.cache_data
def load_store_data():
    return pd.read_csv('../data/air_store_info.csv')

@st.cache_data
def load_stats_data():
    return pd.read_csv('../data/store_stats.csv')

@st.cache_data
def load_visit_data():
    df = pd.read_csv('../data/air_visit_data.csv')
    df['visit_date'] = pd.to_datetime(df['visit_date'])
    return df

model = load_model()
encoder = load_encoder()
store_df = load_store_data()
stats_df = load_stats_data()
visit_df = load_visit_data()

# =========================
# HEADER
# =========================
st.title("🍽️ Restaurant Visitor Forecasting Dashboard")
st.write("Forecast restaurant traffic for any location and date range for better operations and planning.")

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("🔎 Filters")

selected_genre = st.sidebar.selectbox("Filter by Genre", ['All'] + sorted(store_df['air_genre_name'].unique().tolist()))
selected_area = st.sidebar.selectbox("Filter by Area", ['All'] + sorted(store_df['air_area_name'].unique().tolist()))

filtered_df = store_df.copy()
if selected_genre != 'All':
    filtered_df = filtered_df[filtered_df['air_genre_name'] == selected_genre]
if selected_area != 'All':
    filtered_df = filtered_df[filtered_df['air_area_name'] == selected_area]

store_ids = filtered_df['air_store_id'].unique().tolist()
multi_selected_stores = st.sidebar.multiselect("Select Restaurants (Multi-Select)", store_ids, default=store_ids[:1])

# Show selected store info if only one selected
if len(multi_selected_stores) == 1:
    store_info = store_df[store_df['air_store_id'] == multi_selected_stores[0]].iloc[0]
    st.sidebar.markdown(f"📍 **Area**: {store_info['air_area_name']}")
    st.sidebar.markdown(f"🍱 **Genre**: {store_info['air_genre_name']}")

# =========================
# DATE SELECTION
# =========================
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("📅 Start Date", value=datetime.today())
with col2:
    end_date = st.date_input("🗓️ End Date", value=datetime.today())

if end_date < start_date:
    st.error("End date cannot be before start date.")

# =========================
# FEATURE PREPARATION FUNCTION
# =========================
def prepare_features(store_id, dates):
    df = pd.DataFrame({
        'air_store_id': [store_id] * len(dates),
        'visit_date': dates
    })
    df['day_of_week'] = df['visit_date'].dt.dayofweek
    df['year'] = df['visit_date'].dt.year
    df['month'] = df['visit_date'].dt.month
    df['day'] = df['visit_date'].dt.day
    df['week'] = df['visit_date'].dt.isocalendar().week
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df = df.merge(stats_df, on='air_store_id', how='left')
    df['store_id_encoded'] = encoder.transform(df['air_store_id'])
    return df

# =========================
# MULTI-STORE FORECASTING
# =========================
st.markdown("---")
st.subheader("🔮 Forecast for Selected Store(s)")

if st.button("📊 Run Forecast"):
    if end_date < start_date:
        st.error("Please select a valid date range.")
    elif not multi_selected_stores:
        st.error("Please select at least one restaurant.")
    else:
        try:
            date_range = pd.date_range(start=start_date, end=end_date)
            combined_forecasts = pd.DataFrame()

            for store_id in multi_selected_stores:
                df = prepare_features(store_id, date_range)
                features = df[[
                    'year', 'month', 'day', 'day_of_week', 'week', 'is_weekend',
                    'mean_visitors', 'median_visitors', 'min_visitors', 'max_visitors',
                    'std_visitors', 'store_id_encoded'
                ]]
                df['Predicted Visitors'] = model.predict(features).round().astype(int)
                df['visit_date'] = df['visit_date'].dt.date
                df['air_store_id'] = store_id
                combined_forecasts = pd.concat([combined_forecasts, df[['visit_date', 'air_store_id', 'Predicted Visitors']]])

            # Show table
            st.markdown("### 📋 Prediction Table")
            st.dataframe(combined_forecasts.rename(columns={'visit_date': 'Date', 'air_store_id': 'Store ID'}).sort_values(['Store ID', 'Date']), use_container_width=True)

            # Line chart for multiple stores
            pivot = combined_forecasts.pivot(index='visit_date', columns='air_store_id', values='Predicted Visitors')
            st.markdown("### 📈 Prediction Chart")
            st.line_chart(pivot)

        except Exception as e:
            st.error(f"Forecast failed: {e}")

# =========================
# 7-DAY FUTURE FORECAST (Single store if only one selected)
# =========================
if len(multi_selected_stores) == 1:
    st.markdown("---")
    st.subheader("📅 7-Day Future Forecast")

    try:
        next_7_days = pd.date_range(start=datetime.today(), periods=7)
        df_next = prepare_features(multi_selected_stores[0], next_7_days)

        features = df_next[[
            'year', 'month', 'day', 'day_of_week', 'week', 'is_weekend',
            'mean_visitors', 'median_visitors', 'min_visitors', 'max_visitors',
            'std_visitors', 'store_id_encoded'
        ]]
        df_next['Predicted Visitors'] = model.predict(features).round().astype(int)

        st.line_chart(df_next.set_index('visit_date')['Predicted Visitors'])

    except Exception as e:
        st.warning(f"Could not generate 7-day forecast: {e}")

# =========================
# HISTORICAL VISIT TREND (Single store if only one selected)
# =========================
if len(multi_selected_stores) == 1:
    st.markdown("---")
    st.subheader("📈 Historical Visitors Trend")

    try:
        hist_visits = visit_df[visit_df['air_store_id'] == multi_selected_stores[0]]
        hist_visits = hist_visits.sort_values('visit_date')
        hist_visits = hist_visits.set_index('visit_date')

        if not hist_visits.empty:
            st.line_chart(hist_visits['visitors'])
        else:
            st.info("No historical data available.")

    except Exception as e:
        st.warning(f"Failed to load historical data: {e}")

# =========================
# ADVANCED VISUALIZATION: HEATMAP & BOXPLOT (Single store if only one selected)
# =========================
if len(multi_selected_stores) == 1:
    st.markdown("---")
    st.subheader("📊 Heatmap & Boxplot of Historical Visitors")

    try:
        hist_visits = visit_df[visit_df['air_store_id'] == multi_selected_stores[0]].copy()
        hist_visits['visit_date'] = pd.to_datetime(hist_visits['visit_date'])
        hist_visits['day_of_week'] = hist_visits['visit_date'].dt.day_name()
        hist_visits['month'] = hist_visits['visit_date'].dt.strftime('%b')

        # HEATMAP
        pivot_table = hist_visits.pivot_table(index='day_of_week', columns='month', values='visitors', aggfunc='mean')
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_table = pivot_table.reindex(day_order)

        st.markdown("#### 🔥 Average Visitors Heatmap")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap='YlOrRd', ax=ax)
        st.pyplot(fig)

        # BOXPLOT
        st.markdown("#### 📦 Visitors Distribution per Weekday")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=hist_visits, x='day_of_week', y='visitors', order=day_order, palette="Set3", ax=ax2)
        ax2.set_title("Visitors by Day of Week")
        st.pyplot(fig2)

    except Exception as e:
        st.warning(f"Error in visualization: {e}")