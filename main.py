# === Importing Required Libraries === #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, poisson

# === Load Dataset === #
data = pd.read_csv("stolen_vehicles.csv")

# === Basic Exploration === #
print("First 5 rows:\n", data.head())
print("\nData Info:\n")
print(data.info())
print("\nSummary Statistics:\n", data.describe())

# === Handle Missing Values === #
data.fillna(method='ffill', inplace=True)

# === Convert 'STOLEN_DATE' column to datetime === #
data['STOLEN_DATE'] = pd.to_datetime(data['STOLEN_DATE'], errors='coerce')

# === Check Missing Again === #
print("\nMissing Values:\n", data.isnull().sum())

# ==================================================================================
# ✅ Objective 1: Analyze Distribution of Stolen Vehicles Based on Vehicle Type
# ==================================================================================
plt.figure(figsize=(8,6))
vehicle_counts = data['vehicle_type'].value_counts()
sns.barplot(x=vehicle_counts.index, y=vehicle_counts.values, palette='pastel')
plt.title("Most Stolen Vehicle Types")
plt.xticks(rotation=45)
plt.xlabel("Vehicle Type")
plt.ylabel("Number of Thefts")
plt.tight_layout()
plt.show()

# ==================================================================================
# ✅ Objective 2: Most Commonly Stolen Vehicle Colors
# ==================================================================================
plt.figure(figsize=(8,6))
color_counts = data['color'].value_counts()
sns.barplot(x=color_counts.index, y=color_counts.values, palette='muted')
plt.title("Most Common Vehicle Colors Stolen")
plt.xticks(rotation=45)
plt.xlabel("Color")
plt.ylabel("Number of Thefts")
plt.tight_layout()
plt.show()

# ==================================================================================
# ✅ Objective 3: Theft Patterns Over Model Years
# ==================================================================================
plt.figure(figsize=(10,6))
sns.histplot(data['model_year'], bins=20, kde=True, color='skyblue')
plt.title("Theft Distribution over Model Years")
plt.xlabel("Model Year")
plt.ylabel("Number of Thefts")
plt.tight_layout()
plt.show()

# ==================================================================================
# ✅ Objective 4: Location-wise Hotspots (Top 10 Regions)
# ==================================================================================
top_locations = data['Region'].value_counts().head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_locations.values, y=top_locations.index, palette='coolwarm')
plt.title("Top 10 Regions for Vehicle Thefts")
plt.xlabel("Number of Thefts")
plt.ylabel("Region")
plt.tight_layout()
plt.show()

# ==================================================================================
# ✅ Objective 5: Temporal Trends - Date-wise Analysis
# ==================================================================================
date_counts = data['STOLEN_DATE'].value_counts().sort_index()
plt.figure(figsize=(12,6))
date_counts.plot()
plt.title("Vehicle Theft Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Thefts")
plt.tight_layout()
plt.show()

# ==================================================================================
# ✅ Objective 6: Visual Insights & Statistical Analysis on Vehicle Theft Dataset
# ==================================================================================

#  🎨 Optional Styling for Visuals
plt.style.use('seaborn-v0_8-deep')  # or 'ggplot', 'seaborn-whitegrid'

# ==================================================================================
# 🔍 Correlation Analysis
# ==================================================================================
print("\n🔗 Correlation Matrix (Numerical Columns):\n", data.select_dtypes(include='number').corr())

plt.figure(figsize=(10,8))
sns.heatmap(data.select_dtypes(include='number').corr(), 
            annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, square=True,
            cbar_kws={'shrink': 0.75, 'label': 'Correlation Coefficient'})
plt.title("📊 Correlation Heatmap Between Numerical Features", fontsize=16, fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ==================================================================================
# 📦 Outlier Detection using Boxplot (Population Density)
# ==================================================================================
plt.figure(figsize=(8,5))
sns.boxplot(x=data['Density'], color='skyblue')
plt.title("🧪 Outlier Detection using Boxplot - Population Density", fontsize=16, fontweight='bold')
plt.xlabel("Population Density")
plt.tight_layout()
plt.show()

# ==================================================================================
# 📈 ANOVA Test - Comparison of Density across Vehicle Types
# ==================================================================================
grouped_density = [group['Density'].dropna() for name, group in data.groupby('vehicle_type')]
anova_result = stats.f_oneway(*grouped_density)

print("\n📊 ANOVA Test Result (Density across Vehicle Types):")
print(f"   F-Statistic = {anova_result.statistic:.3f}, p-value = {anova_result.pvalue:.3f}")

if anova_result.pvalue < 0.05:
    print("✅ Significant difference in density between vehicle types (p < 0.05)")
else:
    print("❌ No significant difference in density between vehicle types (p >= 0.05)")

# ==================================================================================
# 🔬 Shapiro-Wilk Test for Normality (Density Column)
# ==================================================================================
stat, p = shapiro(data['Density'].dropna())

print("\n🔬 Shapiro-Wilk Normality Test (on Density):")
print(f"   Test Statistic = {stat:.3f}, p-value = {p:.3f}")

if p < 0.05:
    print("❌ Data is not normally distributed (p < 0.05)")
else:
    print("✅ Data seems to be normally distributed (p ≥ 0.05)")

# ==================================================================================
# 📉 Poisson Distribution of Vehicle Thefts Per Day
# ==================================================================================
# Count number of thefts per day
thefts_per_day = data['STOLEN_DATE'].value_counts()
mu = np.mean(thefts_per_day)

# X-axis values and Poisson PMF
x = np.arange(0, max(thefts_per_day)+1)
y = poisson.pmf(x, mu)

plt.figure(figsize=(10,5))
plt.bar(x, y, color='lightcoral', edgecolor='black')
plt.title("📉 Poisson Distribution: Vehicle Thefts Per Day", fontsize=16, fontweight='bold')
plt.xlabel("Number of Thefts per Day")
plt.ylabel("Probability")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()