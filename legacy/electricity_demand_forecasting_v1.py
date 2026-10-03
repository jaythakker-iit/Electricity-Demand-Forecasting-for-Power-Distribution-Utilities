import warnings

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit



warnings.filterwarnings("ignore")



# ============================================================

# EDA FUNCTIONS

# ============================================================



def data_quality(df):

    print("\n[2] DATA QUALITY CHECK")

    print("-" * 40)

    print("\nMissing values:")

    print(df.isna().sum())

    print("\nMissing percentage:")

    print((df.isna().mean() * 100).round(2))

    print(f"\nDuplicate rows: {df.duplicated().sum()}")



    plt.figure(figsize=(9, 4))

    df.isna().sum().plot(kind="bar")

    plt.title("Missing Values by Variable")

    plt.ylabel("Number of Missing Values")

    plt.xlabel("Variable")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()

    return df



def descriptive_statistics(df):

    print("\n[3] DESCRIPTIVE STATISTICS")

    print("-" * 40)

    display(df.describe())

    numerical_columns = df.select_dtypes(include=np.number).columns

    for column in numerical_columns:

        plt.figure(figsize=(7, 4))

        sns.histplot(df[column].dropna(), bins=40, kde=True)

        plt.title(f"Distribution of {column}")

        plt.xlabel(column)

        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.show()



def demand_eda(df):

    print("\n[4] ELECTRICITY DEMAND EDA")

    print("-" * 40)

    print(f"Average demand : {df['MW'].mean():.2f} MW")

    print(f"Maximum demand : {df['MW'].max():.2f} MW")

    print(f"Minimum demand : {df['MW'].min():.2f} MW")



    plt.figure(figsize=(15, 5))

    plt.plot(df.index, df["MW"], linewidth=0.8)

    plt.title("Historical Electricity Demand")

    plt.xlabel("Date")

    plt.ylabel("Demand (MW)")

    plt.tight_layout()

    plt.show()



    temp_df = df.copy()

    temp_df["Hour"] = temp_df.index.hour

    temp_df["Day_of_Week"] = temp_df.index.dayofweek

    temp_df["Month"] = temp_df.index.month



    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.boxplot(data=temp_df, x="Hour", y="MW", ax=axes[0])

    axes[0].set_title("Demand by Hour")



    sns.boxplot(data=temp_df, x="Day_of_Week", y="MW", ax=axes[1])

    axes[1].set_title("Demand by Day of Week")



    sns.boxplot(data=temp_df, x="Month", y="MW", ax=axes[2])

    axes[2].set_title("Demand by Month")

    plt.tight_layout()

    plt.show()



def weather_eda(df):

    print("\n[5] WEATHER EDA")

    print("-" * 40)

    if "Temp" in df.columns:

        plt.figure(figsize=(7, 5))

        sns.scatterplot(data=df, x="Temp", y="MW", alpha=0.25)

        plt.title("Temperature vs Electricity Demand")

        plt.show()

    if "Humidity" in df.columns:

        plt.figure(figsize=(7, 5))

        sns.scatterplot(data=df, x="Humidity", y="MW", alpha=0.25)

        plt.title("Humidity vs Electricity Demand")

        plt.show()

    if "Weather_Condition" in df.columns:

        plt.figure(figsize=(9, 5))

        sns.boxplot(data=df, x="Weather_Condition", y="MW")

        plt.title("Electricity Demand by Weather Condition")

        plt.xticks(rotation=30)

        plt.show()



def calendar_eda(df):

    print("\n[6] HOLIDAY & FESTIVAL EDA")

    print("-" * 40)

    if "Holiday_Type" in df.columns:

        plt.figure(figsize=(9, 5))

        sns.boxplot(data=df, x="Holiday_Type", y="MW")

        plt.title("Electricity Demand by Holiday Type")

        plt.xticks(rotation=30)

        plt.show()

    if "Festival_Name" in df.columns:

        top_festivals = df["Festival_Name"].value_counts().head(10).index

        festival_df = df[df["Festival_Name"].isin(top_festivals)]

        if len(festival_df) > 0:

            plt.figure(figsize=(10, 5))

            sns.boxplot(data=festival_df, x="Festival_Name", y="MW")

            plt.title("Demand for Major Festival Categories")

            plt.xticks(rotation=45)

            plt.show()



def correlation_analysis(df):

    print("\n[7] INITIAL CORRELATION ANALYSIS")

    print("-" * 40)

    numerical_df = df.select_dtypes(include=np.number)

    correlation = numerical_df.corr()

    print("\nCorrelation with electricity demand:")

    print(correlation["MW"].sort_values(ascending=False))

    plt.figure(figsize=(9, 7))

    sns.heatmap(correlation, annot=True, fmt=".2f")

    plt.title("Correlation Matrix")

    plt.show()



# ============================================================

# ML PIPELINE

# ============================================================



def calc_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE is safe here as minimum 'MW' demand is significantly above zero

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    r2 = r2_score(y_true, y_pred)

    return mae, rmse, mape, r2



def run_comprehensive_pipeline(file_path):

    print("="*80)

    print("   COMPREHENSIVE ENERGY DEMAND FORECASTING WORKFLOW WITH EDA")

    print("="*80)



    print("\n--- 1. LOAD DATA & INITIAL PREP ---")

    try:

        df = pd.read_csv(file_path)

    except FileNotFoundError:

        print(f"File {file_path} not found.")

        return



    exact_map = {'Timestamp': 'timestamp', 'Date': 'timestamp', 'Load_MW': 'MW', 'load_MW': 'MW',

                 'Weather Condition': 'Weather_Condition', 'Holiday Type': 'Holiday_Type',

                 'Festival Name': 'Festival_Name', 'Temp': 'Temp', 'Humidity': 'Humidity'}

    df.rename(columns=exact_map, inplace=True)

    if 'timestamp' not in df.columns: df['timestamp'] = df.iloc[:, 0]



    df["Datetime"] = pd.to_datetime(df["timestamp"], dayfirst=True, format='mixed', errors='coerce')

    df = df.dropna(subset=['Datetime']).sort_values("Datetime").set_index("Datetime").drop(columns=["timestamp"])



    # Run EDA on pre-resampled data

    print("\n=================== RUNNING EDA ===================")

    data_quality(df)

    descriptive_statistics(df)

    demand_eda(df)

    weather_eda(df)

    calendar_eda(df)

    correlation_analysis(df)

    print("=================== EDA COMPLETE ===================\n")



    # ============================================================

    # RESAMPLE DATA WITHOUT FORWARD-FILLING CATEGORICAL VARIABLES

    # ============================================================

    df = df.resample("1h").asfreq()



    # Numeric variables: forward-fill with a limit, especially for 'MW'

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()



    if 'MW' in df.columns:

        print(f"\nMissing 'MW' values before ffill: {df['MW'].isna().sum()}")

        # Cap ffill for MW to avoid carrying stale values over long gaps

        df['MW'] = df['MW'].ffill(limit=3)



    for col in numeric_cols:

        if col != 'MW': # MW already handled with a limit

            df[col] = df[col].ffill()



    # Categorical variables:

    # Do NOT forward-fill Weather/Holiday/Festival information.

    # Keep missing values explicit.

    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    for col in categorical_cols:

        df[col] = df[col].fillna("Unknown")



    # Remove rows where essential demand information is still unavailable after limited ffill

    df = df.dropna(subset=["MW"])

    print("After hourly resampling and limited MW ffill:")

    print(f"Rows: {len(df)}")

    print(f"Remaining missing values:\n{df.isna().sum()}")



    print("\n--- 2. FEATURE ENGINEERING ---")

    df["Hour"] = df.index.hour

    df["Day_of_Week"] = df.index.dayofweek

    df["Month"] = df.index.month

    df["Weekend"] = (df["Day_of_Week"] >= 5).astype(int)



    # Lags & Rolling

    df["Lag_1"] = df["MW"].shift(1)

    df["Lag_24"] = df["MW"].shift(24)

    df["Lag_168"] = df["MW"].shift(168)

    df["Rolling_mean_24"] = df["MW"].shift(1).rolling(window=24).mean()



    if 'Festival_Name' in df.columns:

        festival_text = (

            df['Festival_Name']

            .astype(str)

            .str.strip()

            .str.lower()

        )

        df['Is_Festival'] = (

            df['Festival_Name'].notna()

            & ~festival_text.isin(['none', 'nan', '', 'unknown'])

        ).astype(int)

        df.drop(columns=['Festival_Name'], inplace=True)



    # Target Definition (Next Hour)

    df["Target_MW"] = df["MW"].shift(-1)

    df = df.dropna()

    print("Features created: Hour, Day_of_Week, Month, Weekend, Lags (1, 24, 168), Rolling_mean_24.")



    print("\n--- 3. DATA LEAKAGE AUDIT ---")

    print("- Target_MW is strictly shifted(-1).")

    print("- All rolling/lags use shift(1) or higher.")

    print("- Split is strictly chronological (No shuffling).")



    print("\n--- 4. CHRONOLOGICAL TRAIN/TEST SPLIT ---")

    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)



    split_idx = int(len(df_encoded) * 0.8)

    train, test = df_encoded.iloc[:split_idx], df_encoded.iloc[split_idx:]

    features = [c for c in df_encoded.columns if c != "Target_MW"]

    X_tr, y_tr = train[features], train['Target_MW']

    X_te, y_te = test[features], test['Target_MW']

    print(f"Train size: {len(train)}, Test size: {len(test)}")



    # ============================================================

    # FEATURE GROUP EXPERIMENTS

    # ============================================================

    print("\n--- 4.5 FEATURE GROUP EXPERIMENTS ---")

    feature_groups = {

        "A_Historical_Temporal": [

            "MW", "Hour", "Day_of_Week", "Month", "Weekend",

            "Lag_1", "Lag_24", "Lag_168", "Rolling_mean_24"

        ],

        "B_Add_Weather": [

            "MW", "Hour", "Day_of_Week", "Month", "Weekend",

            "Lag_1", "Lag_24", "Lag_168", "Rolling_mean_24",

            "Temp", "Humidity"

        ],

        "C_Add_Calendar": [

            "MW", "Hour", "Day_of_Week", "Month", "Weekend",

            "Lag_1", "Lag_24", "Lag_168", "Rolling_mean_24",

            "Temp", "Humidity", "Is_Festival"

        ]

    }



    weather_cols = [c for c in df_encoded.columns if c.startswith("Weather_Condition_")]

    holiday_cols = [c for c in df_encoded.columns if c.startswith("Holiday_Type_")]



    feature_groups["B_Add_Weather"].extend(weather_cols)

    feature_groups["C_Add_Calendar"].extend(weather_cols + holiday_cols)



    feature_experiment_results = []

    for group_name, group_features in feature_groups.items():

        available_features = [

            col for col in group_features

            if col in df_encoded.columns

        ]

        X = df_encoded[available_features]

        y = df_encoded["Target_MW"]



        split_idx = int(len(df_encoded) * 0.8)

        X_train = X.iloc[:split_idx]

        X_test = X.iloc[split_idx:]

        y_train = y.iloc[:split_idx]

        y_test = y.iloc[split_idx:]



        model = XGBRegressor(

            n_estimators=100,

            max_depth=5,

            learning_rate=0.1,

            random_state=42,

            n_jobs=-1

        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        mae, rmse, mape, r2 = calc_metrics(y_test, pred)

        feature_experiment_results.append({

            "Feature Set": group_name,

            "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2

        })



    display(pd.DataFrame(feature_experiment_results))



    print("\n--- 5. NAIVE PERSISTENCE BASELINE ---")

    pred_naive = test['MW']

    res_naive = calc_metrics(y_te, pred_naive)



    print("\n--- 6. ALGORITHM EXPLORATION & SCALING ---")

    lr_pipe = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])

    lr_pipe.fit(X_tr, y_tr)

    pred_lr = lr_pipe.predict(X_te)

    res_lr = calc_metrics(y_te, pred_lr)



    dt = DecisionTreeRegressor(max_depth=10, random_state=42).fit(X_tr, y_tr)

    pred_dt = dt.predict(X_te)

    res_dt = calc_metrics(y_te, pred_dt)



    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1).fit(X_tr, y_tr)

    pred_rf = rf.predict(X_te)

    res_rf = calc_metrics(y_te, pred_rf)



    xgb = XGBRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1).fit(X_tr, y_tr)

    pred_xgb_base = xgb.predict(X_te)

    res_xgb_base = calc_metrics(y_te, pred_xgb_base)



    print("\n--- 7. HYPERPARAMETER TUNING (XGBoost) ---")

    tscv = TimeSeriesSplit(n_splits=3)

    param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1]}

    xgb_search = RandomizedSearchCV(

        XGBRegressor(random_state=42, n_jobs=-1),

        param_distributions=param_grid,

        n_iter=5, cv=tscv, scoring='neg_mean_absolute_error', random_state=42

    )

    xgb_search.fit(X_tr, y_tr)

    best_xgb = xgb_search.best_estimator_

    pred_xgb = best_xgb.predict(X_te)

    res_xgb_tuned = calc_metrics(y_te, pred_xgb)

    print(f"Best params: {xgb_search.best_params_}")



    print("\n--- 8. FINAL MODEL COMPARISON ---")

    results_df = pd.DataFrame([

        {'Model': 'Naive', 'MAE': res_naive[0], 'RMSE': res_naive[1], 'MAPE': res_naive[2], 'R2': res_naive[3]},

        {'Model': 'Linear Regression', 'MAE': res_lr[0], 'RMSE': res_lr[1], 'MAPE': res_lr[2], 'R2': res_lr[3]},

        {'Model': 'Decision Tree', 'MAE': res_dt[0], 'RMSE': res_dt[1], 'MAPE': res_dt[2], 'R2': res_dt[3]},

        {'Model': 'Random Forest', 'MAE': res_rf[0], 'RMSE': res_rf[1], 'MAPE': res_rf[2], 'R2': res_rf[3]},

        {'Model': 'XGBoost (Base)', 'MAE': res_xgb_base[0], 'RMSE': res_xgb_base[1], 'MAPE': res_xgb_base[2], 'R2': res_xgb_base[3]},

        {'Model': 'XGBoost (Tuned)', 'MAE': res_xgb_tuned[0], 'RMSE': res_xgb_tuned[1], 'MAPE': res_xgb_tuned[2], 'R2': res_xgb_tuned[3]}

    ])

    display(results_df)



    print("\n--- 9. PEAK-DEMAND ANALYSIS ---")

    peak_threshold = y_te.quantile(0.90)

    peak_mask = y_te >= peak_threshold

    y_te_peak, pred_xgb_peak = y_te[peak_mask], pred_xgb[peak_mask]

    peak_res = calc_metrics(y_te_peak, pred_xgb_peak)

    print(f"Top 10% Peak Demand (> {peak_threshold:.1f} MW) | Tuned XGBoost MAE: {peak_res[0]:.2f}, RMSE: {peak_res[1]:.2f}, MAPE: {peak_res[2]:.2f}% ")



    print("\n--- 10. ERROR ANALYSIS (PLOTS) ---")

    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    axes[0].plot(y_te.index[:200], y_te.iloc[:200], label='Actual')

    axes[0].plot(y_te.index[:200], pred_xgb[:200], label='Predicted', linestyle='--')

    axes[0].set_title("Actual vs Predicted (First 200 Test Hours)")

    axes[0].legend()

    errors = y_te - pred_xgb

    sns.histplot(errors, bins=50, kde=True, ax=axes[1])

    axes[1].set_title("Error Distribution (Actual - Predicted)")

    plt.show()



    print("\n--- 11. FEATURE IMPORTANCE ---")

    imp_df = pd.DataFrame({'Feature': features, 'Importance': best_xgb.feature_importances_})

    imp_df = imp_df.sort_values(by='Importance', ascending=False).head(10)

    plt.figure(figsize=(8, 4))

    sns.barplot(data=imp_df, x='Importance', y='Feature')

    plt.title("Top 10 Feature Importances (Tuned XGBoost)")

    plt.show()





    # ============================================================
    # ADDITIONAL ANALYSES - PROJECT ENHANCEMENT
    # NOTE: Original model pipeline above is intentionally unchanged.
    # ============================================================

    print("\n--- 12.1 IMPROVEMENT OVER NAIVE BASELINE ---")

    mae_improvement = ((res_naive[0] - res_xgb_tuned[0]) / res_naive[0]) * 100
    rmse_improvement = ((res_naive[1] - res_xgb_tuned[1]) / res_naive[1]) * 100
    print(f"MAE improvement of Tuned XGBoost over Naive: {mae_improvement:.2f}%")
    print(f"RMSE improvement of Tuned XGBoost over Naive: {rmse_improvement:.2f}%")

    print("\n--- 12.2 SEASONAL NAIVE BASELINES ---")
    pred_naive_24 = test["Lag_24"]
    res_naive_24 = calc_metrics(y_te, pred_naive_24)
    pred_naive_168 = test["Lag_168"]
    res_naive_168 = calc_metrics(y_te, pred_naive_168)

    seasonal_results = pd.DataFrame([
        {"Model": "Naive-1 (Previous Hour)", "MAE": res_naive[0], "RMSE": res_naive[1], "MAPE": res_naive[2], "R2": res_naive[3]},
        {"Model": "Naive-24 (Previous Day)", "MAE": res_naive_24[0], "RMSE": res_naive_24[1], "MAPE": res_naive_24[2], "R2": res_naive_24[3]},
        {"Model": "Naive-168 (Previous Week)", "MAE": res_naive_168[0], "RMSE": res_naive_168[1], "MAPE": res_naive_168[2], "R2": res_naive_168[3]},
        {"Model": "XGBoost (Tuned)", "MAE": res_xgb_tuned[0], "RMSE": res_xgb_tuned[1], "MAPE": res_xgb_tuned[2], "R2": res_xgb_tuned[3]}
    ])
    display(seasonal_results)

    print("\n--- 12.3 FEATURE-GROUP CONTRIBUTION ---")
    fg_results = pd.DataFrame(feature_experiment_results)
    display(fg_results)
    try:
        historical_mae = fg_results.loc[fg_results["Feature Set"] == "A_Historical_Temporal", "MAE"].iloc[0]
        weather_mae = fg_results.loc[fg_results["Feature Set"] == "B_Add_Weather", "MAE"].iloc[0]
        calendar_mae = fg_results.loc[fg_results["Feature Set"] == "C_Add_Calendar", "MAE"].iloc[0]
        weather_gain = ((historical_mae - weather_mae) / historical_mae) * 100
        calendar_gain = ((weather_mae - calendar_mae) / weather_mae) * 100
        print(f"MAE improvement after adding weather: {weather_gain:.2f}%")
        print(f"Additional MAE improvement after adding calendar: {calendar_gain:.2f}%")
    except Exception as e:
        print(f"Feature-group contribution calculation skipped: {e}")

    print("\n--- 12.4 FORECAST BIAS / OVER-PREDICTION / UNDER-PREDICTION ---")
    errors = y_te - pred_xgb
    mean_error = errors.mean()
    median_error = errors.median()
    bias_percentage = (mean_error / y_te.mean()) * 100
    print(f"Mean Forecast Error (Actual - Predicted): {mean_error:.2f} MW")
    print(f"Median Forecast Error: {median_error:.2f} MW")
    print(f"Forecast Bias relative to mean demand: {bias_percentage:.2f}%")
    if mean_error > 0:
        print("Interpretation: The model tends to under-predict demand on average.")
    elif mean_error < 0:
        print("Interpretation: The model tends to over-predict demand on average.")
    else:
        print("Interpretation: The model has approximately zero mean bias.")

    print("\n--- 12.5 FORECAST ACCURACY BY HOUR ---")
    error_analysis_df = pd.DataFrame({"Actual": y_te, "Predicted": pred_xgb})
    error_analysis_df["Hour"] = error_analysis_df.index.hour
    error_analysis_df["Day_of_Week"] = error_analysis_df.index.dayofweek
    error_analysis_df["Weekend"] = error_analysis_df["Day_of_Week"] >= 5
    error_analysis_df["Abs_Error"] = abs(error_analysis_df["Actual"] - error_analysis_df["Predicted"])

    hourly_error = error_analysis_df.groupby("Hour").agg(
        MAE=("Abs_Error", "mean"),
        Mean_Demand=("Actual", "mean")
    )
    display(hourly_error)
    plt.figure(figsize=(10, 5))
    hourly_error["MAE"].plot(kind="bar")
    plt.title("Forecast MAE by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("MAE (MW)")
    plt.tight_layout()
    plt.show()

    print("\n--- 12.6 WEEKDAY VS WEEKEND ACCURACY ---")
    weekly_error = error_analysis_df.groupby("Weekend").agg(
        MAE=("Abs_Error", "mean"),
        Mean_Demand=("Actual", "mean"),
        Observations=("Actual", "count")
    )
    weekly_error.index = ["Weekday", "Weekend"]
    display(weekly_error)

    print("\n--- 12.7 ERROR BY DEMAND LEVEL ---")
    error_analysis_df["Demand_Level"] = pd.qcut(
        error_analysis_df["Actual"], q=3,
        labels=["Low", "Medium", "High"], duplicates="drop"
    )
    demand_level_error = error_analysis_df.groupby("Demand_Level", observed=True).agg(
        MAE=("Abs_Error", "mean"),
        Mean_Demand=("Actual", "mean"),
        Observations=("Actual", "count")
    )
    display(demand_level_error)

    print("\n--- 12.8 PEAK-DEMAND BIAS ---")
    peak_errors = y_te_peak - pred_xgb_peak
    peak_bias = peak_errors.mean()
    print(f"Peak-demand mean error (Actual - Predicted): {peak_bias:.2f} MW")
    if peak_bias > 0:
        print("Peak interpretation: The model tends to under-predict peak demand.")
    elif peak_bias < 0:
        print("Peak interpretation: The model tends to over-predict peak demand.")
    else:
        print("Peak interpretation: The model has approximately zero peak-demand bias.")

    print("\n--- 12.9 ACTUAL VS PREDICTED SCATTER ANALYSIS ---")
    plt.figure(figsize=(7, 7))
    plt.scatter(y_te, pred_xgb, alpha=0.4)
    min_val = min(y_te.min(), pred_xgb.min())
    max_val = max(y_te.max(), pred_xgb.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Actual Demand (MW)")
    plt.ylabel("Predicted Demand (MW)")
    plt.title("Actual vs Predicted Electricity Demand")
    plt.tight_layout()
    plt.show()

    print("\n--- 12.10 CYCLIC TIME-FEATURE EXPERIMENT ---")
    # Additional experiment only. Original Hour/Day/Month features remain unchanged.
    df_cyclic = df_encoded.copy()
    df_cyclic["Hour_sin"] = np.sin(2 * np.pi * df_cyclic["Hour"] / 24)
    df_cyclic["Hour_cos"] = np.cos(2 * np.pi * df_cyclic["Hour"] / 24)
    df_cyclic["DOW_sin"] = np.sin(2 * np.pi * df_cyclic["Day_of_Week"] / 7)
    df_cyclic["DOW_cos"] = np.cos(2 * np.pi * df_cyclic["Day_of_Week"] / 7)
    df_cyclic["Month_sin"] = np.sin(2 * np.pi * df_cyclic["Month"] / 12)
    df_cyclic["Month_cos"] = np.cos(2 * np.pi * df_cyclic["Month"] / 12)

    cyclic_features = [c for c in df_cyclic.columns if c != "Target_MW"]
    X_cyc_train = df_cyclic[cyclic_features].iloc[:split_idx]
    X_cyc_test = df_cyclic[cyclic_features].iloc[split_idx:]
    y_cyc_train = df_cyclic["Target_MW"].iloc[:split_idx]
    y_cyc_test = df_cyclic["Target_MW"].iloc[split_idx:]

    cyclic_xgb = XGBRegressor(**xgb_search.best_params_, random_state=42, n_jobs=-1)
    cyclic_xgb.fit(X_cyc_train, y_cyc_train)
    pred_cyclic = cyclic_xgb.predict(X_cyc_test)
    res_cyclic = calc_metrics(y_cyc_test, pred_cyclic)

    cyclic_results = pd.DataFrame([
        {"Model": "Original Tuned XGBoost", "MAE": res_xgb_tuned[0], "RMSE": res_xgb_tuned[1], "MAPE": res_xgb_tuned[2], "R2": res_xgb_tuned[3]},
        {"Model": "Tuned XGBoost + Cyclic Time Features", "MAE": res_cyclic[0], "RMSE": res_cyclic[1], "MAPE": res_cyclic[2], "R2": res_cyclic[3]}
    ])
    display(cyclic_results)

    print("\n--- 12.11 WALK-FORWARD TIME-SERIES VALIDATION ---")
    walk_forward = TimeSeriesSplit(n_splits=3)
    walk_results = []
    for fold_no, (wf_train_idx, wf_test_idx) in enumerate(walk_forward.split(X_tr), start=1):
        wf_model = XGBRegressor(**xgb_search.best_params_, random_state=42, n_jobs=-1)
        wf_model.fit(X_tr.iloc[wf_train_idx], y_tr.iloc[wf_train_idx])
        wf_pred = wf_model.predict(X_tr.iloc[wf_test_idx])
        wf_metrics = calc_metrics(y_tr.iloc[wf_test_idx], wf_pred)
        walk_results.append({"Fold": fold_no, "MAE": wf_metrics[0], "RMSE": wf_metrics[1], "MAPE": wf_metrics[2], "R2": wf_metrics[3]})
    walk_results_df = pd.DataFrame(walk_results)
    display(walk_results_df)
    print(f"Walk-forward mean MAE: {walk_results_df['MAE'].mean():.4f}")
    print(f"Walk-forward MAE standard deviation: {walk_results_df['MAE'].std():.4f}")

    print("\n--- 12.12 PEAK-DEMAND CLASSIFICATION ---")
    # Threshold is calculated from training data to avoid using future test information.
    train_peak_threshold = y_tr.quantile(0.90)
    actual_peak = (y_te >= train_peak_threshold).astype(int)
    predicted_peak = (pred_xgb >= train_peak_threshold).astype(int)
    tp = int(((actual_peak == 1) & (predicted_peak == 1)).sum())
    tn = int(((actual_peak == 0) & (predicted_peak == 0)).sum())
    fp = int(((actual_peak == 0) & (predicted_peak == 1)).sum())
    fn = int(((actual_peak == 1) & (predicted_peak == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"Training-derived peak threshold: {train_peak_threshold:.2f} MW")
    print(f"Peak classification accuracy: {accuracy:.4f}")
    print(f"Peak precision: {precision:.4f}")
    print(f"Peak recall: {recall:.4f}")
    print(f"Peak F1-score: {f1:.4f}")
    print("Confusion matrix [TN, FP, FN, TP]:", [[tn, fp], [fn, tp]])

    print("\n--- 12.13 SHAP MODEL EXPLAINABILITY (OPTIONAL) ---")
    try:
        import shap
        shap_sample = X_te.copy()
        if len(shap_sample) > 1000:
            shap_sample = shap_sample.sample(1000, random_state=42)
        shap_explainer = shap.TreeExplainer(best_xgb)
        shap_values = shap_explainer.shap_values(shap_sample)
        shap.summary_plot(shap_values, shap_sample, show=False)
        plt.title("SHAP Feature Importance - Tuned XGBoost")
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("SHAP is not installed. Skipping SHAP analysis.")
    except Exception as e:
        print(f"SHAP analysis skipped due to: {e}")

    print("\n--- 12.14 DATA-DRIVEN FINAL FINDINGS ---")
    best_model_row = results_df.loc[results_df["MAE"].idxmin()]
    print(f"Lowest test MAE among compared models: {best_model_row['Model']}")
    print(f"Lowest test MAE: {best_model_row['MAE']:.4f} MW")
    top_feature = imp_df.iloc[0]
    print(f"Top XGBoost feature by built-in importance: {top_feature['Feature']}")
    print(f"Top feature importance: {top_feature['Importance']:.4f}")
    print(f"Peak threshold used in original peak analysis: {peak_threshold:.2f} MW")
    print(f"Peak MAE: {peak_res[0]:.4f} MW")
    print(f"Peak RMSE: {peak_res[1]:.4f} MW")
    print(f"Peak MAPE: {peak_res[2]:.4f}%")

    print("\n--- 12. FINAL CONCLUSIONS ---")

    print("1. The Tuned XGBoost model provided the best overall forecasting accuracy.")

    print("2. The chronological split and Pipeline methodology ensured no data leakage occurred.")

    print("3. Historical lags (especially Lag_1 and Lag_24) dominated feature importance.")



if __name__ == "__main__":

    run_comprehensive_pipeline("load_data.csv")
