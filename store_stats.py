import pandas as pd

# Load raw visit data
visit_df = pd.read_csv('C:/Users/macke/Desktop/study_project/my_projects/restaurant_demand_forecasting/data/feature_engineered_data.csv')

# Convert visit_date to datetime
visit_df['visit_date'] = pd.to_datetime(visit_df['visit_date'])

# Group by store to compute basic stats
stats_df = visit_df.groupby('air_store_id')['visitors'].agg(
    mean_visitors='mean',
    median_visitors='median',
    min_visitors='min',
    max_visitors='max',
    std_visitors='std'
).reset_index()

# Save to CSV
stats_df.to_csv('C:/Users/macke/Desktop/study_project/my_projects/restaurant_demand_forecasting/data/store_stats.csv', index=False)

print("store_stats.csv has been generated and saved in /data/")
