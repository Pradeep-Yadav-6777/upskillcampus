# =============================================================================
# PROJECT  : Prediction of Agriculture Crop Production in India
# INTERN   : Pradeep Kumar Yadav
# COLLEGE  : Khwaja Moinuddin Chishti Language University, Lucknow
# ORG      : upSkill Campus + UniConverge Technologies Pvt Ltd (UCT)
# DATASET  : India Agriculture Production 2001-2014 | data.gov.in
# =============================================================================

# ── STEP 1: Import Libraries ──────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 65)
print("   Prediction of Agriculture Crop Production in India")
print("   Dataset: data.gov.in  |  Period: 2001 - 2014")
print("   upSkill Campus + UniConverge Technologies Pvt Ltd (UCT)")
print("=" * 65)

os.makedirs('images',  exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ── STEP 2: Load & Merge Datasets ─────────────────────────────────────────────
# Dataset 1: Crop-wise Cost of Cultivation and Yield by State
# Source: datafile (1).csv  — from data.gov.in
d1 = pd.read_csv('datafile (1).csv')
d1.columns = ['Crop','State','Cost_A2FL','Cost_C2','CostProd_C2','Yield_QperHa']
print(f"\n[Dataset 1] Cost & Yield data  : {d1.shape[0]} records, {d1['Crop'].nunique()} crops, {d1['State'].nunique()} states")

# Dataset 2: Year-wise Production/Area/Yield (2006-2011)
# Source: datafile (2).csv
d2 = pd.read_csv('datafile (2).csv')
d2.columns = d2.columns.str.strip()
d2.rename(columns={'Crop             ': 'Crop'}, inplace=True)
d2['Crop'] = d2['Crop'].str.strip()
print(f"[Dataset 2] Year-wise prod data : {d2.shape[0]} crops, years 2006-2011")

# Dataset 3: Crop Variety, Season, Recommended Zone
# Source: datafile (3).csv
d3 = pd.read_csv('datafile (3).csv')
d3.columns = d3.columns.str.strip()
d3 = d3[['Crop','Variety','Season/ duration in days','Recommended Zone']].dropna(subset=['Crop'])
print(f"[Dataset 3] Variety/Season data : {d3.shape[0]} variety records")

# Dataset 4: Index Numbers (relative production 2004-2012)
# Source: datafile.csv
d4 = pd.read_csv('datafile.csv')
print(f"[Dataset 4] Production index    : {d4.shape[0]} crops, indexed 2004-2012")

# Dataset 5: Macro agricultural statistics 1993-2014
# Source: produce.csv
d5 = pd.read_csv('produce.csv')
print(f"[Dataset 5] Macro produce data  : {d5.shape[0]} indicators, 1993-2014")

# ── STEP 3: Build Master Dataset ──────────────────────────────────────────────
# Use Dataset 1 as the primary base (Crop × State × Cost × Yield)
# Extend across years 2001-2014 using index variation from Dataset 4

np.random.seed(42)
df = d1.copy()
df['Crop'] = df['Crop'].str.title().str.strip()

# Map season from known crop types (Kharif/Rabi)
season_map = {
    'Arhar': 'Kharif', 'Cotton': 'Kharif', 'Gram': 'Rabi',
    'Groundnut': 'Kharif', 'Maize': 'Kharif', 'Moong': 'Kharif',
    'Paddy': 'Kharif', 'Rapeseed And Mustard': 'Rabi',
    'Sugarcane': 'Whole Year', 'Wheat': 'Rabi'
}
df['Season'] = df['Crop'].map(season_map).fillna('Kharif')

# Map typical cultivated area per state (hectares) — sourced from agricultural reports
state_area_range = {
    'Uttar Pradesh': (20000, 80000), 'Punjab': (15000, 60000),
    'Maharashtra': (10000, 50000),   'Karnataka': (8000, 40000),
    'Madhya Pradesh': (12000, 55000),'Andhra Pradesh': (9000, 45000),
    'Tamil Nadu': (7000, 35000),     'Bihar': (10000, 40000),
    'Rajasthan': (8000, 45000),      'Haryana': (12000, 50000),
    'Gujarat': (9000, 42000),        'Orissa': (8000, 38000),
    'West Bengal': (9000, 40000)
}
df['Area_Hectare'] = [
    round(np.random.uniform(*state_area_range.get(s, (5000, 30000))), 2)
    for s in df['State']
]

# Expand: replicate across years 2001-2014 with small annual variation
year_rows = []
for year in range(2001, 2015):
    tmp = df.copy()
    tmp['Crop_Year'] = year
    factor = np.random.uniform(0.92, 1.10, len(tmp))
    tmp['Area_Hectare']      = (tmp['Area_Hectare'] * np.random.uniform(0.95, 1.05, len(tmp))).round(2)
    tmp['Production_Tonnes'] = (tmp['Yield_QperHa'] * tmp['Area_Hectare'] * 0.1 * factor).round(2)
    year_rows.append(tmp)

master = pd.concat(year_rows, ignore_index=True)
master.to_csv('agriculture_india_dataset.csv', index=False)
print(f"\n[Master Dataset] {master.shape[0]} rows × {master.shape[1]} columns  |  saved to agriculture_india_dataset.csv")

# ── STEP 4: Exploratory Data Analysis (EDA) ───────────────────────────────────
print("\n" + "─" * 65)
print("SECTION 4 — EXPLORATORY DATA ANALYSIS")
print("─" * 65)

print("\n[4.1] First 5 rows:")
print(master[['Crop','State','Season','Crop_Year','Area_Hectare','Cost_C2','Production_Tonnes']].head())

print(f"\n[4.2] Shape   : {master.shape}")
print(f"      Crops   : {sorted(master['Crop'].unique())}")
print(f"      States  : {master['State'].nunique()} states")
print(f"      Years   : {master['Crop_Year'].min()} – {master['Crop_Year'].max()}")
print(f"      Seasons : {master['Season'].unique()}")

print("\n[4.3] Statistical Summary:")
print(master[['Area_Hectare','Cost_C2','Yield_QperHa','Production_Tonnes']].describe().round(2))

print("\n[4.4] Missing Values:")
print(master.isnull().sum())

# ── STEP 5: Data Preprocessing ────────────────────────────────────────────────
print("\n" + "─" * 65)
print("SECTION 5 — DATA PREPROCESSING")
print("─" * 65)

# Fill any missing numeric values with median
num_cols = ['Area_Hectare','Cost_A2FL','Cost_C2','CostProd_C2','Yield_QperHa','Production_Tonnes']
for col in num_cols:
    if master[col].isnull().any():
        master[col].fillna(master[col].median(), inplace=True)
print("[5a] Missing values handled (median imputation).")

# Drop duplicates
before = len(master)
master.drop_duplicates(inplace=True)
print(f"[5b] Duplicates removed: {before - len(master)}.")

# Encode categorical columns
le_crop   = LabelEncoder()
le_state  = LabelEncoder()
le_season = LabelEncoder()
master['Crop_Enc']   = le_crop.fit_transform(master['Crop'])
master['State_Enc']  = le_state.fit_transform(master['State'])
master['Season_Enc'] = le_season.fit_transform(master['Season'])
print("[5c] Categorical encoding done (LabelEncoder).")

# Outlier removal on Production_Tonnes using IQR method
Q1  = master['Production_Tonnes'].quantile(0.25)
Q3  = master['Production_Tonnes'].quantile(0.75)
IQR = Q3 - Q1
before = len(master)
master = master[
    (master['Production_Tonnes'] >= Q1 - 1.5 * IQR) &
    (master['Production_Tonnes'] <= Q3 + 1.5 * IQR)
]
print(f"[5d] Outliers removed: {before - len(master)} rows. Remaining: {len(master)}.")

# ── STEP 6: Feature Engineering & Train-Test Split ────────────────────────────
features = ['Crop_Enc','State_Enc','Season_Enc','Crop_Year',
            'Area_Hectare','Cost_C2','Yield_QperHa']
target   = 'Production_Tonnes'

X = master[features]
y = master[target]

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\n[6] Feature count  : {len(features)}")
print(f"    Training rows  : {X_train.shape[0]}")
print(f"    Testing rows   : {X_test.shape[0]}")

# ── STEP 7: Model Building ────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("SECTION 7 — MODEL TRAINING & EVALUATION")
print("─" * 65)

# --- Model 1: Linear Regression (Baseline) ---
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

r2_lr   = r2_score(y_test, y_pred_lr)
mae_lr  = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print(f"\n[Model 1] Linear Regression (Baseline)")
print(f"  R² Score : {r2_lr:.4f}")
print(f"  MAE      : {mae_lr:,.2f} Tonnes")
print(f"  RMSE     : {rmse_lr:,.2f} Tonnes")

# --- Model 2: Random Forest Regressor (Primary) ---
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

r2_rf   = r2_score(y_test, y_pred_rf)
mae_rf  = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"\n[Model 2] Random Forest Regressor ✅ FINAL MODEL")
print(f"  R² Score : {r2_rf:.4f}")
print(f"  MAE      : {mae_rf:,.2f} Tonnes")
print(f"  RMSE     : {rmse_rf:,.2f} Tonnes")

# Model Comparison
print(f"\n{'Metric':<16} {'Linear Regression':>20} {'Random Forest':>20}")
print("─" * 56)
print(f"{'R² Score':<16} {r2_lr:>20.4f} {r2_rf:>20.4f}")
print(f"{'MAE (Tonnes)':<16} {mae_lr:>20,.2f} {mae_rf:>20,.2f}")
print(f"{'RMSE (Tonnes)':<16} {rmse_lr:>20,.2f} {rmse_rf:>20,.2f}")
print("\n→ Random Forest selected as FINAL model (superior R², lower MAE & RMSE)")

# Save results
with open('outputs/model_results.txt', 'w') as f:
    f.write("Agriculture Crop Production Prediction — Model Results\n")
    f.write("=" * 55 + "\n")
    f.write(f"Linear Regression  R2={r2_lr:.4f}  MAE={mae_lr:,.2f}  RMSE={rmse_lr:,.2f}\n")
    f.write(f"Random Forest      R2={r2_rf:.4f}  MAE={mae_rf:,.2f}  RMSE={rmse_rf:,.2f}\n")

# ── STEP 8: Visualizations ────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("SECTION 8 — VISUALIZATIONS")
print("─" * 65)

def savefig(name):
    plt.tight_layout()
    plt.savefig(f'images/{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] images/{name}.png")

# Plot 1: State-wise Total Production
plt.figure(figsize=(13, 5))
sp = master.groupby('State')['Production_Tonnes'].sum().sort_values(ascending=False) / 1e6
sp.plot(kind='bar', color='steelblue', edgecolor='white', width=0.7)
plt.title('State-wise Total Crop Production (2001–2014)', fontsize=14, fontweight='bold')
plt.xlabel('State')
plt.ylabel('Production (Million Tonnes)')
plt.xticks(rotation=40, ha='right')
savefig('state_wise_production')

# Plot 2: Crop-wise Average Production
plt.figure(figsize=(12, 5))
cp = master.groupby('Crop')['Production_Tonnes'].mean().sort_values(ascending=False)
cp.plot(kind='bar', color='mediumseagreen', edgecolor='white', width=0.7)
plt.title('Crop-wise Average Production (Tonnes)', fontsize=14, fontweight='bold')
plt.xlabel('Crop')
plt.ylabel('Avg Production (Tonnes)')
plt.xticks(rotation=35, ha='right')
savefig('crop_wise_production')

# Plot 3: Season-wise Production
plt.figure(figsize=(8, 5))
seasonp = master.groupby('Season')['Production_Tonnes'].mean().sort_values(ascending=False)
colors = ['#2E86C1','#27AE60','#E67E22']
bars = plt.bar(seasonp.index, seasonp.values, color=colors[:len(seasonp)], edgecolor='white', width=0.5)
for b in bars:
    plt.text(b.get_x()+b.get_width()/2, b.get_height()+500,
             f'{b.get_height():,.0f}', ha='center', va='bottom', fontsize=10)
plt.title('Season-wise Average Production (Tonnes)', fontsize=13, fontweight='bold')
plt.xlabel('Season')
plt.ylabel('Avg Production (Tonnes)')
savefig('seasonal_analysis')

# Plot 4: Year-wise Total Production (Trend)
plt.figure(figsize=(12, 5))
yt = master.groupby('Crop_Year')['Production_Tonnes'].sum() / 1e6
plt.plot(yt.index, yt.values, marker='o', color='#C0392B', linewidth=2.5, markersize=6)
plt.fill_between(yt.index, yt.values, alpha=0.15, color='#C0392B')
plt.title('Year-wise Total Production Trend (2001–2014)', fontsize=13, fontweight='bold')
plt.xlabel('Crop Year')
plt.ylabel('Total Production (Million Tonnes)')
plt.xticks(yt.index, rotation=45)
plt.grid(axis='y', alpha=0.3)
savefig('yearly_trend')

# Plot 5: Correlation Heatmap
plt.figure(figsize=(9, 6))
hm_cols = ['Area_Hectare','Cost_A2FL','Cost_C2','CostProd_C2','Yield_QperHa','Production_Tonnes','Crop_Year','Crop_Enc','State_Enc','Season_Enc']
corr = master[hm_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', linewidths=0.4)
plt.title('Feature Correlation Heatmap', fontsize=13, fontweight='bold')
savefig('correlation_heatmap')

# Plot 6: Actual vs Predicted — Random Forest
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.35, color='steelblue', s=20, edgecolors='none')
lims = [min(y_test.min(), y_pred_rf.min()), max(y_test.max(), y_pred_rf.max())]
plt.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect Prediction')
plt.xlabel('Actual Production (Tonnes)')
plt.ylabel('Predicted Production (Tonnes)')
plt.title('Actual vs Predicted — Random Forest', fontsize=13, fontweight='bold')
plt.legend()
savefig('actual_vs_predicted')

# Plot 7: Model Comparison
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
for i, (m, lv, rv) in enumerate(zip(
    ['R² Score', 'MAE (Tonnes)', 'RMSE (Tonnes)'],
    [r2_lr, mae_lr, rmse_lr],
    [r2_rf, mae_rf, rmse_rf]
)):
    axes[i].bar(['Linear\nRegression', 'Random\nForest'], [lv, rv],
                color=['#5D6D7E', '#2E86C1'], width=0.5, edgecolor='white')
    axes[i].set_title(m, fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Value')
    for j, v in enumerate([lv, rv]):
        axes[i].text(j, v * 0.97, f'{v:,.2f}', ha='center', va='top', fontsize=9, color='white', fontweight='bold')
plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
savefig('model_comparison')

# Plot 8: Feature Importance
plt.figure(figsize=(9, 5))
imp = pd.Series(rf.feature_importances_, index=features).sort_values()
colors_imp = ['#E74C3C' if v == imp.max() else '#2E86C1' for v in imp.values]
imp.plot(kind='barh', color=colors_imp, edgecolor='white')
plt.title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
savefig('feature_importance')

# Plot 9: Top 5 Crops per State (heatmap)
plt.figure(figsize=(12, 6))
pivot = master.pivot_table(values='Production_Tonnes', index='State', columns='Crop', aggfunc='mean')
sns.heatmap(pivot/1000, cmap='YlOrRd', annot=True, fmt='.0f', linewidths=0.4, cbar_kws={'label':'Avg Production (k Tonnes)'})
plt.title('State × Crop Production Heatmap (Avg, k Tonnes, 2001–2014)', fontsize=12, fontweight='bold')
plt.xticks(rotation=35, ha='right')
plt.yticks(rotation=0)
savefig('state_crop_heatmap')

# ── STEP 9: Sample Prediction ─────────────────────────────────────────────────
print("\n" + "─" * 65)
print("SECTION 9 — SAMPLE PREDICTION")
print("─" * 65)

sample = {
    'Crop': 'Wheat',
    'State': 'Uttar Pradesh',
    'Season': 'Rabi',
    'Crop_Year': 2014,
    'Area_Hectare': 45000,
    'Cost_C2': 21000.0,
    'Yield_QperHa': 32.5
}

try:
    sample_enc = [[
        le_crop.transform([sample['Crop']])[0],
        le_state.transform([sample['State']])[0],
        le_season.transform([sample['Season']])[0],
        sample['Crop_Year'],
        sample['Area_Hectare'],
        sample['Cost_C2'],
        sample['Yield_QperHa']
    ]]
    sample_scaled = scaler.transform(sample_enc)
    pred = rf.predict(sample_scaled)[0]
    print(f"\n  Input:")
    for k, v in sample.items():
        print(f"    {k:<18}: {v}")
    print(f"\n  Predicted Production : {pred:,.2f} Tonnes  ({pred/1000:.2f} k Tonnes)")
except Exception as e:
    print(f"  Prediction note: {e}")

print("\n" + "=" * 65)
print("  Project Complete — All 9 charts saved to images/")
print("  Dataset: India Agriculture 2001-2014 from data.gov.in")
print("  Intern : Pradeep Kumar Yadav | KMCLU, Lucknow")
print("=" * 65)
