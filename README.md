# Valorant VCT Winner Prediction System

A comprehensive machine learning system for predicting Valorant Champions Tour (VCT) match outcomes using real-time esports data from the Valorant Esports API.

## 🎯 Features

- **Real-time Data Collection**: Fetches live match data from Valorant Esports APIs
- **Advanced Feature Engineering**: Creates sophisticated features from team statistics, regional strength, and head-to-head records
- **Multiple ML Models**: Implements Random Forest, Gradient Boosting, Logistic Regression, and SVM
- **Comprehensive Evaluation**: Detailed performance metrics, visualizations, and cross-validation
- **Prediction Interface**: Easy-to-use interface for making match predictions
- **Interactive Analysis**: Jupyter notebook for data exploration and model analysis

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd valorant-ml
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Run the complete pipeline**:
```bash
python main.py --mode full
```

2. **Make a prediction**:
```bash
python main.py --mode predict --model-path valorant_model_random_forest --team1 "G2 Esports" --team2 "Team Heretics" --country1 "us" --country2 "eu"
```

3. **Interactive analysis**:
```bash
jupyter notebook valorant_analysis.ipynb
```

## 📊 Data Sources

The system uses two main APIs:

1. **VLR Esports API** (https://vlresports.vercel.app/): Team, player, event, and match data
2. **VLR Results API** (https://vlr.orlandomm.net/api/v1/results): Match results and statistics

## 🏗️ System Architecture

### Core Components

1. **Data Collection** (`data_collector.py`)
   - Fetches data from Valorant Esports APIs
   - Handles rate limiting and error management
   - Saves raw data for offline processing

2. **Data Preprocessing** (`data_preprocessor.py`)
   - Cleans and structures match data
   - Calculates team statistics and regional strength
   - Handles missing data and outliers

3. **Feature Engineering** (`feature_engineering.py`)
   - Creates ML-ready features from raw data
   - Implements advanced features like momentum and strength of schedule
   - Handles feature scaling and normalization

4. **Machine Learning** (`ml_model.py`)
   - Implements multiple ML algorithms
   - Performs hyperparameter tuning
   - Provides model selection and evaluation

5. **Evaluation** (`evaluation.py`)
   - Comprehensive model evaluation metrics
   - Visualization tools for performance analysis
   - Cross-validation and learning curve analysis

6. **Prediction Interface** (`prediction_interface.py`)
   - User-friendly interface for making predictions
   - Team comparison and analysis tools
   - Batch prediction capabilities

## 🔧 Configuration

Edit `config.py` to customize:

- API endpoints and rate limiting
- Model parameters and feature selection
- Data collection limits and thresholds

## 📈 Model Performance

The system evaluates models using multiple metrics:

- **Accuracy**: Overall prediction accuracy
- **Precision/Recall**: Balanced performance metrics
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **AUC-PR**: Area under the precision-recall curve

## 🎮 Features Used for Prediction

### Team Statistics
- Overall win rate
- Recent form (last 10 matches)
- Average score per match
- Score difference vs opponents

### Regional Analysis
- Regional strength based on team performance
- Regional win rates and trends

### Head-to-Head Records
- Historical matchups between teams
- Advantage calculations

### Advanced Features
- Momentum (performance in recent matches)
- Strength of schedule
- Tournament experience
- Match recency and importance

## 📊 Visualizations

The system generates comprehensive visualizations:

- ROC curves and Precision-Recall curves
- Confusion matrices with percentages
- Feature importance analysis
- Learning curves for overfitting detection
- Prediction confidence distributions
- Cross-validation results

## 🚀 Usage Examples

### Training a New Model

```python
from data_collector import ValorantDataCollector
from data_preprocessor import ValorantDataPreprocessor
from feature_engineering import ValorantFeatureEngineer
from ml_model import ValorantMLModel

# Collect data
collector = ValorantDataCollector()
raw_data = collector.collect_all_data()

# Preprocess data
preprocessor = ValorantDataPreprocessor()
df, team_stats, region_strength, h2h_stats = preprocessor.preprocess_data(raw_data)

# Engineer features
feature_engineer = ValorantFeatureEngineer(team_stats, region_strength, h2h_stats)
X, y, metadata, feature_names = feature_engineer.prepare_ml_features(df)

# Train model
ml_model = ValorantMLModel()
results = ml_model.train_full_pipeline(X, y, feature_names)
```

### Making Predictions

```python
from prediction_interface import ValorantPredictionInterface

# Load trained model
predictor = ValorantPredictionInterface("valorant_model_random_forest")

# Make prediction
prediction = predictor.predict_match("G2 Esports", "Team Heretics", "us", "eu")
print(f"Predicted Winner: {prediction['predicted_winner']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

### Team Analysis

```python
# Get team analysis
team_analysis = predictor.get_team_analysis("G2 Esports")
print(team_analysis)

# Compare teams
comparison = predictor.compare_teams("G2 Esports", "Team Heretics")
print(comparison)
```

## 📁 File Structure

```
valorant-ml/
├── main.py                    # Main execution script
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data_collector.py          # Data collection module
├── data_preprocessor.py       # Data preprocessing module
├── feature_engineering.py     # Feature engineering module
├── ml_model.py               # Machine learning models
├── evaluation.py             # Model evaluation framework
├── prediction_interface.py   # Prediction interface
└── valorant_analysis.ipynb   # Interactive analysis notebook
```

## 🔄 Workflow

1. **Data Collection**: Fetch live data from APIs
2. **Preprocessing**: Clean and structure the data
3. **Feature Engineering**: Create ML-ready features
4. **Model Training**: Train multiple ML algorithms
5. **Evaluation**: Assess model performance
6. **Prediction**: Make match outcome predictions

## 🛠️ Dependencies

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- requests
- jupyter

## 📝 Notes

- The system is designed to work with the Valorant Esports API
- Model performance depends on data quality and quantity
- Regular retraining with new data is recommended
- The system handles missing data gracefully with fallback values

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Valorant Esports API providers
- VLR.gg for comprehensive esports data
- The Valorant esports community for inspiration
