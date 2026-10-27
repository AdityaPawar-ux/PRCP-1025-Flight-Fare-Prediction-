# Flight Fare Prediction (PRCP-1025)

A machine learning regression project that estimates **flight ticket prices (in INR)** from
historical flight information. It compares multiple models, recommends the best one for
production, and includes a journey planner that helps customers compare flights and pick
cheaper dates.

> The model gives an **estimate** based on historical patterns in the supplied dataset
> (flights from March to June 2019). It is not a real-time airline fare quote.

## Problem Statement
Flight prices are hard to guess: the same flight can cost a different amount tomorrow. This
project analyzes flight-fare data and builds a model that predicts ticket prices, helping
customers plan their journey and helping airlines understand which prices the market supports.

## Project Tasks
1. Complete exploratory data analysis (EDA) of the flight-fare dataset
2. Predictive model for flight prices that helps customers plan their journey
3. Model comparison report with a production recommendation
4. Report on data challenges and the techniques used to address them

## Dataset
- **File:** `Flight_Fare.xlsx`
- **Size:** 10,683 flights, 11 columns (10,461 flights after cleaning)
- **Period:** 1 March 2019 to 27 June 2019
- **Columns:** Airline, Date_of_Journey, Source, Destination, Route, Dep_Time, Arrival_Time,
  Duration, Total_Stops, Additional_Info, Price (target)

## Approach
1. **Data cleaning:** removed 1 incomplete row, 220 duplicate rows and 1 impossible record
   (a 2-stop flight listed as 5 minutes); merged "New Delhi" into "Delhi" and "No Info"
   into "No info"
2. **Feature extraction:** converted text dates, times and durations into numeric features
   (journey day/month/weekday, departure and arrival hour/minute, duration in minutes,
   number of stops)
3. **EDA:** price distribution, airline, stops, source city, route, month, weekday, departure
   time, duration, additional info, correlations and outliers, with written insights
4. **Feature selection:** dropped redundant columns (`Route` repeats `Total_Stops`;
   `Destination` is determined by `Source`) and avoided target leakage
5. **Modelling:** 9 regression models inside scikit-learn pipelines (one-hot encoding,
   scaling where needed), trained on log(price) and scored in rupees
6. **Tuning:** RandomizedSearchCV with 5-fold cross-validation
7. **Extra checks:** feature importance, error analysis by airline, and a future-month test
   (train on March to May, test on June)
8. **Using the model:** a fare estimator, a journey planner (ranks flights by predicted fare
   for a chosen date) and a price calendar (cheapest dates for a given flight)

## Results
| Model | CV R² | Test R² | RMSE (₹) | MAE (₹) |
|---|---|---|---|---|
| **Tuned XGBoost** | **0.903** | **0.918** | **1,305** | **645** |
| XGBoost | 0.893 | 0.915 | 1,329 | 726 |
| Tuned Gradient Boosting | 0.891 | 0.908 | 1,377 | 741 |
| Tuned Random Forest | 0.870 | 0.907 | 1,389 | 662 |
| Random Forest | 0.870 | 0.903 | 1,415 | 645 |
| SVR | 0.847 | 0.868 | 1,655 | 1,002 |
| Decision Tree | 0.819 | 0.851 | 1,758 | 742 |
| KNN | 0.772 | 0.799 | 2,039 | 1,135 |
| Linear Regression | 0.654 | 0.643 | 2,717 | 1,754 |

**Best model: Tuned XGBoost.** It has the highest cross-validated accuracy and the lowest
rupee error. About **77%** of predictions are within 10% of the real fare and **92%** are
within 20%.

**Future-month check (train March to May, test June):** R² of about **0.85** for the tuned
Random Forest and XGBoost, so the model generalizes to an unseen month, with somewhat lower
accuracy than the random-split score.

## Key Findings
- **Airline** is the strongest price driver, followed by **flight duration**, journey date,
  `Additional_Info` and number of stops
- Low-cost carriers (SpiceJet, IndiGo, GoAir, Air Asia) cost about half of full-service
  airlines; Jet Airways Business is about 5x higher
- Non-stop flights average about ₹5,000 versus about ₹10,600 for 1-stop flights
- Delhi → Cochin is the busiest route (about 42% of flights) and the most expensive
- April looks cheapest only because 64% of April flights are non-stop (about 30% in other
  months), which is a mix effect and not a real seasonal discount
- Linear models are weak (R² about 0.64) because fares depend on interactions between airline,
  stops and route, which tree ensembles capture well
- Price is right-skewed (skewness 1.86), so models were trained on log(price)

## Limitations
- Only four months of data, so the model cannot be trusted for other seasons or holidays
- No booking date (days before departure), which strongly affects real fares
- Rare categories (business class, Trujet, premium economy) are predicted less reliably
- The journey planner and price calendar show patterns learned from historical data, not
  guaranteed future prices

## Repository Contents
| File | Description |
|---|---|
| `PRCP-1025-FlightPricePrediction.ipynb` | Full analysis: cleaning, EDA, modelling, comparison, challenges report, planner |
| `Flight_Fare.xlsx` | Dataset |
| `flight_fare_model.pkl` | Saved best model (optional) |
| `requirements.txt` | Python packages needed |
| `README.md` | This file |

## How to Run
```bash
git clone https://github.com/<your-username>/FlightFarePrediction.git
cd FlightFarePrediction
pip install -r requirements.txt
jupyter notebook PRCP-1025-FlightPricePrediction.ipynb
```
The notebook takes about 5 to 10 minutes to run end to end because of hyperparameter tuning.

## Using the Saved Model
```python
import joblib
model = joblib.load("flight_fare_model.pkl")
```
Load it with the same scikit-learn version it was trained with.

## Tech Stack
Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, openpyxl, joblib
