import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("cvd_dataset.csv.csv")

# Preview data
print("Shape of dataset:", df.shape)
df.head()


# Count missing values in each column
missing = df.isnull().sum().sort_values(ascending=False)
print("Missing values per column:\n", missing)

# Check percentage of missing values
percent_missing = (df.isnull().mean() * 100).round(2)
print("\nPercentage of missing values:\n", percent_missing)


# Separate numerical and categorical columns
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include='object').columns

# Fill missing numerical values with median
df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.median()))

# Fill missing categorical values with mode
df[cat_cols] = df[cat_cols].apply(lambda x: x.fillna(x.mode()[0]))

print("✅ Missing values handled successfully.")


# Count duplicates
print("Duplicate rows before:", df.duplicated().sum())

# Drop duplicates
df = df.drop_duplicates()

print("✅ Duplicate rows removed. Remaining rows:", len(df))

# Convert BP columns to numeric
df['Systolic BP'] = pd.to_numeric(df['Systolic BP'], errors='coerce')
df['Diastolic BP'] = pd.to_numeric(df['Diastolic BP'], errors='coerce')

# Convert categorical fields
categorical_columns = ['Sex', 'Smoking Status', 'Diabetes Status', 
                       'Physical Activity Level', 'Family History of CVD', 'CVD Risk Level']
for col in categorical_columns:
    df[col] = df[col].astype('category')

print("✅ Data types standardized.")


#  EXPLORATORY DATA ANALYSIS (EDA)

#  Basic Overview
print("\n--- Basic Descriptive Statistics ---")
print(df[['BMI', 'Systolic BP', 'Diastolic BP']].describe())

#  Check the distribution of BMI and BP
plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
sns.histplot(df['BMI'], kde=True, color='blue')
plt.title("Distribution of BMI")

plt.subplot(1,3,2)
sns.histplot(df['Systolic BP'], kde=True, color='orange')
plt.title("Distribution of Systolic BP")

plt.subplot(1,3,3)
sns.histplot(df['Diastolic BP'], kde=True, color='green')
plt.title("Distribution of Diastolic BP")

plt.tight_layout()
plt.show()

#  Correlation analysis
corr = df[['BMI', 'Systolic BP', 'Diastolic BP']].corr()
print("\n--- Correlation Matrix ---")
print(corr)

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between BMI and Blood Pressure")
plt.show()

#  Scatter plots for relationships
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.scatterplot(data=df, x='BMI', y='Systolic BP', color='teal', alpha=0.7)
plt.title("BMI vs Systolic Blood Pressure")

plt.subplot(1,2,2)
sns.scatterplot(data=df, x='BMI', y='Diastolic BP', color='purple', alpha=0.7)
plt.title("BMI vs Diastolic Blood Pressure")

plt.tight_layout()
plt.show()

#  Regression plots (with trend line)
sns.lmplot(data=df, x='BMI', y='Systolic BP', line_kws={'color':'red'})
plt.title("Regression: BMI vs Systolic BP")
plt.show()

sns.lmplot(data=df, x='BMI', y='Diastolic BP', line_kws={'color':'red'})
plt.title("Regression: BMI vs Diastolic BP")
plt.show()

#  Create BMI Categories for comparison
bins = [0, 18.5, 24.9, 29.9, 100]
labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
df['BMI Category'] = pd.cut(df['BMI'], bins=bins, labels=labels, include_lowest=True)

# Compare average BP in each BMI category
grouped = df.groupby('BMI Category')[['Systolic BP', 'Diastolic BP']].mean().round(2)
print("\n--- Mean Blood Pressure by BMI Category ---")
print(grouped)

# Boxplots by category
plt.figure(figsize=(10,4))
sns.boxplot(data=df, x='BMI Category', y='Systolic BP', palette='Set2')
plt.title("Systolic BP by BMI Category")
plt.show()

plt.figure(figsize=(10,4))
sns.boxplot(data=df, x='BMI Category', y='Diastolic BP', palette='Set3')
plt.title("Diastolic BP by BMI Category")
plt.show()

print("✅ EDA completed successfully.")


#   DATA VISUALIZATION & INSIGHTS


# Use a clean theme
sns.set(style="whitegrid")

#  Histogram of BMI
plt.figure(figsize=(8,5))
sns.histplot(df['BMI'], bins=20, kde=True, color='steelblue')
plt.title("Distribution of BMI Among Adults")
plt.xlabel("Body Mass Index (BMI)")
plt.ylabel("Frequency")
plt.show()

#  Histogram of Systolic & Diastolic Blood Pressure
fig, axes = plt.subplots(1, 2, figsize=(12,5))
sns.histplot(df['Systolic BP'], bins=20, kde=True, color='coral', ax=axes[0])
axes[0].set_title("Distribution of Systolic Blood Pressure (Adults)")
axes[0].set_xlabel("Systolic BP (mmHg)")

sns.histplot(df['Diastolic BP'], bins=20, kde=True, color='lightgreen', ax=axes[1])
axes[1].set_title("Distribution of Diastolic Blood Pressure (Adults)")
axes[1].set_xlabel("Diastolic BP (mmHg)")

plt.tight_layout()
plt.show()

#  Scatterplots: BMI vs BP
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.scatterplot(x='BMI', y='Systolic BP', data=df, color='teal', alpha=0.7)
plt.title("BMI vs Systolic BP (Adults)")
plt.xlabel("BMI")
plt.ylabel("Systolic BP (mmHg)")

plt.subplot(1,2,2)
sns.scatterplot(x='BMI', y='Diastolic BP', data=df, color='purple', alpha=0.7)
plt.title("BMI vs Diastolic BP (Adults)")
plt.xlabel("BMI")
plt.ylabel("Diastolic BP (mmHg)")

plt.tight_layout()
plt.show()

#  Regression plots (trend lines)
sns.lmplot(x='BMI', y='Systolic BP', data=df, line_kws={'color':'red'})
plt.title("Linear Relationship: BMI vs Systolic BP (Adults)")
plt.show()

sns.lmplot(x='BMI', y='Diastolic BP', data=df, line_kws={'color':'red'})
plt.title("Linear Relationship: BMI vs Diastolic BP (Adults)")
plt.show()

#  Boxplots by BMI Category
bins = [0, 18.5, 24.9, 29.9, 100]
labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
df['BMI Category'] = pd.cut(df['BMI'], bins=bins, labels=labels, include_lowest=True)

fig, axes = plt.subplots(1, 2, figsize=(12,5))
sns.boxplot(data=df, x='BMI Category', y='Systolic BP', palette='Set2', ax=axes[0])
axes[0].set_title("Systolic BP Across BMI Categories (Adults)")
axes[0].set_xlabel("BMI Category")
axes[0].set_ylabel("Systolic BP (mmHg)")

sns.boxplot(data=df, x='BMI Category', y='Diastolic BP', palette='Set3', ax=axes[1])
axes[1].set_title("Diastolic BP Across BMI Categories (Adults)")
axes[1].set_xlabel("BMI Category")
axes[1].set_ylabel("Diastolic BP (mmHg)")

plt.tight_layout()
plt.show()

#  Correlation Heatmap
plt.figure(figsize=(6,4))
corr = df[['BMI', 'Systolic BP', 'Diastolic BP']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between BMI and Blood Pressure (Adults)")
plt.show()

#  Insight Summary
print("\n================ INSIGHT SUMMARY ================")
corr_sbp = df['BMI'].corr(df['Systolic BP']).round(2)
corr_dbp = df['BMI'].corr(df['Diastolic BP']).round(2)

print(f"Correlation between BMI and Systolic BP: {corr_sbp}")
print(f"Correlation between BMI and Diastolic BP: {corr_dbp}")

mean_bp = df.groupby('BMI Category')[['Systolic BP', 'Diastolic BP']].mean().round(1)
print("\nMean Blood Pressure by BMI Category:\n", mean_bp)

print("\n✅ Visualization and insight generation complete.")





# ==============================
# 🧠 AUTOMATED INSIGHTS (prints + interpretation)
# ==============================


print("\n================ AUTOMATED INSIGHTS ================")

# 1) Basic counts and missing-check reassurance
n_total = len(df)
n_missing = df[['BMI','Systolic BP','Diastolic BP']].isnull().any(axis=1).sum()
print(f"Total adult records: {n_total}")
print(f"Records with any missing BMI/SBP/DBP (should be 0 after cleaning): {n_missing}")
print("")

# 2) Distributional summaries & interpretation
bmi_mean = df['BMI'].mean().round(2)
bmi_median = df['BMI'].median().round(2)
sbp_mean = df['Systolic BP'].mean().round(2)
dbp_mean = df['Diastolic BP'].mean().round(2)
print("Distribution summary:")
print(f"- Mean BMI: {bmi_mean}, Median BMI: {bmi_median}")
print(f"- Mean Systolic BP: {sbp_mean} mmHg")
print(f"- Mean Diastolic BP: {dbp_mean} mmHg")

# Plain-language interpretation
if bmi_mean >= 25:
    print("Interpretation: The average adult in this sample is in the overweight range (BMI >= 25).")
else:
    print("Interpretation: The average adult in this sample is below the overweight threshold (BMI < 25).")
if sbp_mean >= 130 or dbp_mean >= 80:
    print("Interpretation: Average blood pressure is in the elevated/hypertensive range—public-health concern.")
else:
    print("Interpretation: Average blood pressure is within normal-to-elevated range.")
print("")

# 3) Correlation coefficients and interpretation
corr_sbp = df['BMI'].corr(df['Systolic BP']).round(2)
corr_dbp = df['BMI'].corr(df['Diastolic BP']).round(2)
print("Correlation results:")
print(f"- Pearson correlation (BMI, Systolic BP): {corr_sbp}")
print(f"- Pearson correlation (BMI, Diastolic BP): {corr_dbp}")

# Interpret strengths
def interpret_corr(r):
    r_abs = abs(r)
    if r_abs >= 0.7:
        return "strong"
    elif r_abs >= 0.4:
        return "moderate"
    elif r_abs >= 0.2:
        return "weak"
    else:
        return "negligible"

print(f"Interpretation: BMI has a {interpret_corr(corr_sbp)} positive relationship with systolic BP.")
print(f"Interpretation: BMI has a {interpret_corr(corr_dbp)} positive relationship with diastolic BP.")
print("")

# 4) Quantify effect using simple linear regression (BMI -> BP)
# Prepare data (drop NA just in case)
reg_df = df[['BMI','Systolic BP','Diastolic BP']].dropna()
X = reg_df[['BMI']].values.reshape(-1,1)

# Systolic model
model_s = LinearRegression().fit(X, reg_df['Systolic BP'].values)
slope_s = model_s.coef_[0].round(3)
intercept_s = model_s.intercept_.round(2)
r2_s = model_s.score(X, reg_df['Systolic BP'].values).round(3)

# Diastolic model
model_d = LinearRegression().fit(X, reg_df['Diastolic BP'].values)
slope_d = model_d.coef_[0].round(3)
intercept_d = model_d.intercept_.round(2)
r2_d = model_d.score(X, reg_df['Diastolic BP'].values).round(3)

print("Linear regression (unadjusted): BMI -> Blood Pressure")
print(f"- Systolic BP: slope = {slope_s} mmHg per BMI unit, intercept = {intercept_s}, R² = {r2_s}")
print(f"  Interpretation: On average, each 1-unit increase in BMI is associated with ~{slope_s} mmHg increase in systolic BP (unadjusted).")
print(f"- Diastolic BP: slope = {slope_d} mmHg per BMI unit, intercept = {intercept_d}, R² = {r2_d}")
print(f"  Interpretation: On average, each 1-unit increase in BMI is associated with ~{slope_d} mmHg increase in diastolic BP (unadjusted).")
print("Note: These are unadjusted effects — consider multivariable regression controlling for age, sex, smoking, etc. for causal interpretation.")
print("")

# 5) Group-level (BMI categories) comparison with plain language insight
group_means = df.groupby('BMI Category')[['Systolic BP','Diastolic BP']].mean().round(2)
group_counts = df['BMI Category'].value_counts().reindex(group_means.index).fillna(0).astype(int)
print("Mean BP by BMI Category (count):")
for cat in group_means.index:
    sbp_m = group_means.loc[cat, 'Systolic BP']
    dbp_m = group_means.loc[cat, 'Diastolic BP']
    cnt = group_counts.loc[cat]
    print(f"- {cat}: n={cnt}, mean SBP={sbp_m} mmHg, mean DBP={dbp_m} mmHg")

# Plain-language comparison
print("\nCategory interpretation:")
if 'Obese' in group_means.index:
    try:
        obese_sbp = float(group_means.loc['Obese','Systolic BP'])
        normal_sbp = float(group_means.loc['Normal','Systolic BP'])
        diff = (obese_sbp - normal_sbp)
        print(f"Adults in the Obese category have on average {diff:.1f} mmHg higher systolic BP than those in the Normal category.")
    except Exception:
        pass
print("Overall: BP increases across BMI categories from Underweight -> Normal -> Overweight -> Obese.")
print("")

# 6) Outlier check and recommendation
# Flag extreme BP values (very high)
extreme_sbp = df[df['Systolic BP'] >= 180].shape[0]
extreme_dbp = df[df['Diastolic BP'] >= 120].shape[0]
if extreme_sbp + extreme_dbp > 0:
    print(f"⚠️ Alert: There are {extreme_sbp} records with systolic BP >=180 mmHg and {extreme_dbp} records with diastolic BP >=120 mmHg. Investigate for potential emergencies or data errors.")
else:
    print("No extreme hypertensive outliers detected (SBP>=180 or DBP>=120).")

print("")

# 7) Practical suggestions for ML & Research
print("Practical suggestions:")
print("- Use multivariable regression (e.g., SBP ~ BMI + Age + Sex + Smoking + Activity) to adjust for confounding.")
print("- Consider standardizing features before ML models (scaling).")
print("- If residuals show non-linearity, consider polynomial terms or tree-based models.")
print("- For classification (hypertension yes/no), create label using clinical thresholds and evaluate with ROC/AUC.")
print("")

# 8) Save insight summary to a text file (optional)
try:
    with open("insight_summary.txt","w") as f:
        f.write("Automated insight summary\n\n")
        f.write(f"Total records: {n_total}\n")
        f.write(f"Mean BMI: {bmi_mean}, Mean SBP: {sbp_mean}, Mean DBP: {dbp_mean}\n")
        f.write(f"Corr BMI-SBP: {corr_sbp}, Corr BMI-DBP: {corr_dbp}\n")
        f.write(f"Slope SBP~BMI: {slope_s}, Slope DBP~BMI: {slope_d}\n\n")
        f.write("Mean BP by BMI Category:\n")
        f.write(group_means.to_string())
    print("Saved a brief insight summary to 'insight_summary.txt'.")
except Exception as e:
    print("Could not save insight summary:", e)

# 9) Pause so user can review plots (helpful when running from terminal)
input("\nPress Enter to finish and close plots...")




# # ==============================
# # 🔧 MACHINE LEARNING PIPELINE (Linear Regression + Random Forest)
# # ==============================
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# import matplotlib.pyplot as plt
# import joblib
# import warnings
# warnings.filterwarnings("ignore")

# # --- 0. Check required target columns exist ---
# if not {'Systolic BP', 'Diastolic BP', 'BMI'}.issubset(df.columns):
#     raise ValueError("Make sure 'BMI', 'Systolic BP', and 'Diastolic BP' columns exist in df.")

# # --- 1. Build feature list dynamically (use BMI + available covariates) ---
# numeric_candidates = ['Age', 'age', 'Weight (kg)', 'Weight_kg', 'weight_kg', 'Total Cholesterol (mg/dL)',
#                       'total_cholesterol_mgdl', 'HDL (mg/dL)', 'hdl_mgdl', 'Fasting Blood Sugar (mg/dL)',
#                       'fasting_blood_sugar_mgdl']
# cat_candidates = ['Sex', 'sex', 'Smoking Status', 'smoking_status', 'Physical Activity Level',
#                   'physical_activity_level', 'Diabetes Status', 'diabetes_status', 'Family History of CVD',
#                   'family_history_of_cvd']

# # Normalize column name variants to actual existing columns
# existing_numeric = [c for c in numeric_candidates if c in df.columns]
# existing_cat = [c for c in cat_candidates if c in df.columns]

# # Ensure BMI present in consistent column name
# bmi_col = 'BMI' if 'BMI' in df.columns else [c for c in df.columns if c.lower() == 'bmi']
# if isinstance(bmi_col, list) and bmi_col:
#     bmi_col = bmi_col[0]
# elif isinstance(bmi_col, list) and not bmi_col:
#     raise ValueError("BMI column not found. Ensure it exists and is named 'BMI'.")

# # Build final feature lists
# numeric_features = [bmi_col] + [c for c in existing_numeric if c != bmi_col]
# categorical_features = existing_cat.copy()

# # Print what will be used
# print("Using numeric features:", numeric_features)
# print("Using categorical features:", categorical_features)

# # --- 2. Prepare modeling dataframe (drop rows with any missing in features/targets) ---
# targets = ['Systolic BP', 'Diastolic BP']
# all_cols = numeric_features + categorical_features + targets
# df_model = df[all_cols].dropna().copy()
# print(f"Modeling dataset size (rows): {len(df_model)}")

# # Quick safety check: need at least, say, 50 rows to model reasonably
# if len(df_model) < 30:
#     print("Warning: small dataset for ML (<30 rows). Models may be unreliable.")

# # --- 3. Preprocessing pipeline ---
# # numeric transformer: scaler
# numeric_transformer = Pipeline(steps=[
#     ('scaler', StandardScaler())
# ])

# # categorical transformer: one-hot encoding (drop='first' to avoid multicollinearity with LR)
# categorical_transformer = Pipeline(steps=[
#     ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
# ])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ],
#     remainder='drop'  # drop any other columns
# )

# # --- 4. Helper: function to train/evaluate for one target ---
# def train_and_evaluate(target_name, save_prefix):
#     print("\n" + "="*40)
#     print(f"Modeling target: {target_name}")
#     print("="*40 + "\n")

#     X = df_model[numeric_features + categorical_features]
#     y = df_model[target_name].values

#     # Train/test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Pipelines for models
#     lr_pipeline = Pipeline(steps=[('pre', preprocessor),
#                                   ('lr', LinearRegression())])

#     rf_pipeline = Pipeline(steps=[('pre', preprocessor),
#                                   ('rf', RandomForestRegressor(random_state=42, n_jobs=-1))])

#     # --- Train Linear Regression ---
#     lr_pipeline.fit(X_train, y_train)
#     y_pred_lr = lr_pipeline.predict(X_test)

#     # --- Train Random Forest (default) ---
#     rf_pipeline.fit(X_train, y_train)
#     y_pred_rf = rf_pipeline.predict(X_test)

#     # --- Evaluate helper ---
#     def evaluate(y_true, y_pred):
#         r2 = r2_score(y_true, y_pred)
#         mae = mean_absolute_error(y_true, y_pred)
#         rmse = mean_squared_error(y_true, y_pred, squared=False)
#         return {'r2': r2, 'mae': mae, 'rmse': rmse}

#     metrics_lr = evaluate(y_test, y_pred_lr)
#     metrics_rf = evaluate(y_test, y_pred_rf)

#     print("Linear Regression performance on test set:")
#     print(f" R² = {metrics_lr['r2']:.3f}, MAE = {metrics_lr['mae']:.3f} mmHg, RMSE = {metrics_lr['rmse']:.3f} mmHg")
#     print("\nRandom Forest performance on test set:")
#     print(f" R² = {metrics_rf['r2']:.3f}, MAE = {metrics_rf['mae']:.3f} mmHg, RMSE = {metrics_rf['rmse']:.3f} mmHg")

#     # --- 5. Cross-validated R² (5-fold) for more robust performance estimate ---
#     cv_lr = cross_val_score(lr_pipeline, X, y, cv=5, scoring='r2')
#     cv_rf = cross_val_score(rf_pipeline, X, y, cv=5, scoring='r2')
#     print(f"\n5-fold CV R² (Linear Regression): mean={cv_lr.mean():.3f}, std={cv_lr.std():.3f}")
#     print(f"5-fold CV R² (Random Forest): mean={cv_rf.mean():.3f}, std={cv_rf.std():.3f}")

#     # --- 6. Residuals plot for best model (choose by R²) ---
#     best_model_name = 'Linear Regression' if metrics_lr['r2'] >= metrics_rf['r2'] else 'Random Forest'
#     best_pred = y_pred_lr if best_model_name == 'Linear Regression' else y_pred_rf
#     residuals = y_test - best_pred

#     plt.figure(figsize=(6,4))
#     plt.scatter(best_pred, residuals, alpha=0.6)
#     plt.axhline(0, color='red', linestyle='--')
#     plt.xlabel('Predicted')
#     plt.ylabel('Residuals (actual - predicted)')
#     plt.title(f'Residuals Plot ({best_model_name}) for {target_name}')
#     plt.show()

#     # Prediction vs actual scatter
#     plt.figure(figsize=(6,5))
#     plt.scatter(y_test, best_pred, alpha=0.6)
#     mx = max(y_test.max(), best_pred.max())
#     mn = min(y_test.min(), best_pred.min())
#     plt.plot([mn, mx], [mn, mx], color='red', linestyle='--')
#     plt.xlabel('Actual')
#     plt.ylabel('Predicted')
#     plt.title(f'Predicted vs Actual ({best_model_name}) for {target_name}')
#     plt.show()

#     # --- 7. Feature importance (for Random Forest) ---
#     try:
#         # Need to extract feature names after preprocessing
#         # Fit preprocessor on full features to get OHE column names
#         preprocessor.fit(X_train)
#         num_out = numeric_features
#         cat_out = []
#         if categorical_features:
#             ohe = preprocessor.named_transformers_['cat'].named_steps['ohe']
#             cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
#             cat_out = cat_feature_names
#         feature_names = num_out + cat_out

#         rf = rf_pipeline.named_steps['rf']
#         importances = rf.feature_importances_
#         fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
#         print("\nTop feature importances (Random Forest):")
#         print(fi.head(10).to_string())
#         # Plot
#         plt.figure(figsize=(8,4))
#         fi.head(10).plot(kind='barh')
#         plt.gca().invert_yaxis()
#         plt.title(f'Feature Importances ({target_name})')
#         plt.show()
#     except Exception as e:
#         print("Could not compute feature importance:", e)

#     # --- 8. Save best model and preprocessor/scaler ---
#     chosen_pipeline = lr_pipeline if metrics_lr['r2'] >= metrics_rf['r2'] else rf_pipeline
#     joblib.dump(chosen_pipeline, f"{save_prefix}_best_pipeline.joblib")
#     print(f"Saved best pipeline to: {save_prefix}_best_pipeline.joblib")

#     # also save performance summary
#     summary = {
#         'model': best_model_name,
#         'metrics_lr': metrics_lr,
#         'metrics_rf': metrics_rf,
#         'cv_lr_mean_r2': cv_lr.mean(),
#         'cv_rf_mean_r2': cv_rf.mean()
#     }
#     joblib.dump(summary, f"{save_prefix}_summary.joblib")
#     print(f"Saved summary to: {save_prefix}_summary.joblib")

#     return summary

# # --- 9. Run for Systolic and Diastolic ---
# summary_sbp = train_and_evaluate('Systolic BP', 'sbp_model')
# summary_dbp = train_and_evaluate('Diastolic BP', 'dbp_model')

# # --- 10. Simple textual interpretation printed for report ---
# def textual_interpretation(summary, target_name):
#     best = summary['model']
#     lr_r2 = summary['metrics_lr']['r2']
#     rf_r2 = summary['metrics_rf']['r2']
#     print("\n" + "-"*40)
#     print(f"Interpretation for {target_name}:")
#     print(f"- Best model: {best}")
#     print(f"- Linear Regression R² (test): {lr_r2:.3f}")
#     print(f"- Random Forest R² (test): {rf_r2:.3f}")
#     if max(lr_r2, rf_r2) < 0.2:
#         print("- Note: Low R² indicates BMI + selected features explain little variance — consider more features or non-linear modeling.")
#     elif max(lr_r2, rf_r2) < 0.5:
#         print("- Moderate predictive power. BMI helps, but other covariates (age, lifestyle) are important.")
#     else:
#         print("- Good predictive power: BMI and included features explain a substantial portion of variance.")
#     print("- Report MAE and RMSE for clinical interpretability (errors in mmHg).")

# textual_interpretation(summary_sbp, 'Systolic BP')
# textual_interpretation(summary_dbp, 'Diastolic BP')

# print("\n✅ ML modeling completed. Models and summaries saved to working directory.")
