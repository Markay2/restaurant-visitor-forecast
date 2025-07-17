# 🍽️ Restaurant Visitor Forecast

An AI-powered dashboard for predicting restaurant visitor traffic using machine learning. This application helps restaurant owners and managers make data-driven decisions about staffing, inventory, and operations.

### Live Demo

**[View Live Application](https://restaurant-visitor-forecast-qhqjd2mmnej8ed4jjxc3yx.streamlit.app/)**

### Features

## Intelligent Forecasting
- AI-powered visitor predictions using XGBoost model
- Customizable forecast periods (7 days, 30 days, or custom range)
- Real-time predictions with confidence intervals

## Advanced Analytics
- Historical visitor pattern analysis
- Interactive heatmaps showing peak days and months
- Weekend vs weekday performance comparison
- Multi-store performance comparison

## Smart Filtering
- Filter restaurants by genre and location
- Search functionality for easy restaurant selection
- Store-specific performance metrics

## Interactive Visualizations
- Dynamic charts with Plotly
- Downloadable forecast reports (CSV)
- Responsive dashboard design
- Modern UI with gradient styling

### Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: XGBoost, scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Deployment**: Streamlit Cloud

### Requirements

```python
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
seaborn
matplotlib
plotly
```

### Installation & Setup

  1. Clone the Repository
  bash
git clone https://github.com/yourusername/restaurant-visitor-forecast.git
cd restaurant-visitor-forecast


 2. Install Dependencies
  bash
pip install -r requirements.txt


 3. Prepare Data Files
Ensure you have the following data files in the `data/` directory:
- `air_store_info.csv` - Restaurant information
- `air_visit_data.csv` - Historical visitor data
- `store_stats.csv` - Store performance statistics

 4. Prepare Model Files
Place trained models in the `models/` directory:
- `xgb_visitor_model.pkl` - XGBoost forecasting model
- `store_label_encoder.pkl` - Store ID encoder

 5. Run the Application
  bash
streamlit run app.py


## 📁 Project Structure

```
restaurant-visitor-forecast/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore file
├── data/                 # Data files
│   ├── air_store_info.csv
│   ├── air_visit_data.csv
│   └── store_stats.csv
├── models/               # Trained ML models
│   ├── xgb_visitor_model.pkl
│   └── store_label_encoder.pkl
└── notebooks/            # Jupyter notebooks (if any)
    └── model_training.ipynb
```

### Configuration

## Environment Variables
Create a `.env` file for any sensitive configurations:

# Add any API keys or sensitive data here


## Model Configuration
The application uses pre-trained models. To retrain:
1. Prepare your training data
2. Run the training pipeline
3. Save models to the `models/` directory

### Data Requirements

## Store Information (`air_store_info.csv`)
- `air_store_id`: Unique store identifier
- `air_area_name`: Restaurant area/location
- `air_genre_name`: Restaurant genre/category

## Visit Data (`air_visit_data.csv`)
- `air_store_id`: Store identifier
- `visit_date`: Date of visit
- `visitors`: Number of visitors

## Store Statistics (`store_stats.csv`)
- `air_store_id`: Store identifier
- `mean_visitors`: Average daily visitors
- `median_visitors`: Median daily visitors
- `min_visitors`: Minimum daily visitors
- `max_visitors`: Maximum daily visitors
- `std_visitors`: Standard deviation of visitors

### Features Overview

## Dashboard Metrics
- Total restaurants in database
- Available genres
- Historical visit statistics
- Data coverage period

## Forecasting Controls
- Genre and area filtering
- Store selection with search
- Flexible date range selection
- Multi-store comparison mode

## Visualization Types
- **Line Charts**: Historical trends and forecasts
- **Heatmaps**: Visitor patterns by day/month
- **Bar Charts**: Multi-store comparisons
- **Metrics Cards**: Key performance indicators

### How It Works

1. **Data Loading**: Application loads historical data and trained models
2. **Store Selection**: Users filter and select restaurants
3. **Feature Engineering**: Date features and store statistics are prepared
4. **Prediction**: XGBoost model generates visitor forecasts
5. **Visualization**: Interactive charts display results
6. **Export**: Users can download forecasts as CSV

### Deployment

## Streamlit Cloud
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Configure deployment settings
4. Deploy automatically

## Local Development
 bash
# Run in development mode
streamlit run app.py --server.runOnSave true


### Performance Metrics

The forecasting model provides:
- **Accuracy**: Based on historical validation
- **Speed**: Real-time predictions
- **Scalability**: Handles multiple stores
- **Reliability**: Error handling and validation

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request


###  Support

For questions or issues:
- Create an issue on GitHub
- Contact: [your-email@example.com]

### Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Machine Learning with [XGBoost](https://xgboost.readthedocs.io/)
- Visualizations powered by [Plotly](https://plotly.com/)



Made with ❤️ for the restaurant industry