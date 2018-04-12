# Singapore-Housing-Price-Prediction

Codebase of Singapore Housing Price Prediction Competition on Kaggle (In-Class), final projects of CS5339 Machine Learning at National University of Singapore (NUS).

## Models
Stacked Ensembled Models, using Lasso, Elastic Net, Gradient Boosting, XGBoost and LightGBM.

## Data
Please get your data from [Kaggle](https://www.kaggle.com/c/singapore-housing-prices) and put them under `data` directory.

## Usage
```
pip install -r requirements.txt

# For HDB flat training & prediction
python HDB.py

# For private houses training & prediction
python Private.py
```

## Credits
- [Comprehensive data exploration with Python](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)
- [Stacked Regressions : Top 4% on LeaderBoard](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard)
