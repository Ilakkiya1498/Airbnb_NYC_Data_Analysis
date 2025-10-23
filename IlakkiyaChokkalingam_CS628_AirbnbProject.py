# -----------------------------------------------
# Project name: CS628 – Monroe University: Airbnb NYC Data Analysis 
# Author: Ilakkiya Chokkalingam
# Submission Date: July 10,2025
# -----------------------------------------------

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway, pearsonr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------------
# 1. Load and Clean Dataset
# --------------------------------------------------------------------------------

# Load dataset
# Used pandas to read the CSV file and load it into a DataFrame.
df_raw = pd.read_csv("AB_NYC_2019.csv")
df = df_raw.copy()
sns.set(style="whitegrid")


# 1.1 Remove exact duplicate rows
dups_before = df.duplicated().sum()
print(f"Duplicate rows before cleaning: {dups_before}")
df.drop_duplicates(inplace=True)
print(f"Duplicate rows after cleaning: {df.duplicated().sum()}")

# 1.2 Check for and handle missing values
print("Missing values per column:\n", df.isnull().sum())
df.dropna(subset=["price", "neighbourhood_group", "room_type"], inplace=True)

# 1.3 Convert types and clean numeric columns
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["minimum_nights"] = pd.to_numeric(df["minimum_nights"], errors="coerce")

# 1.4 Remove invalid or extreme values
# Basic cleaning: removed listings with 0 or unrealistic high prices
# These are likely errors or luxury listings skewing analysis.
df = df[df["price"] > 0]
df = df[df["price"] < 1000]
df = df[df["minimum_nights"] >= 1]

# 1.5 Identified price outliers via IQR method
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f"Price outlier bounds: {lower_bound:.2f} to {upper_bound:.2f}")

outliers = df[(df["price"] < lower_bound) | (df["price"] > upper_bound)]
print(f"Number of price outliers: {len(outliers)}")


# --------------------------------------------------------------------------------
# 2. Average Price by Borough (with ANOVA Test)
# --------------------------------------------------------------------------------

# Group by borough to calculate mean prices
borough_avg = df.groupby("neighbourhood_group")["price"].mean()
print("\n[2] Average Price by Borough:\n", borough_avg)

# Performed ANOVA test to check if price differences across boroughs are statistically significant
boroughs = df["neighbourhood_group"].unique()
groups = [df[df["neighbourhood_group"] == b]["price"] for b in boroughs]
anova_result = f_oneway(*groups)
print(f"ANOVA p-value: {anova_result.pvalue:.6f}")

alpha = 0.05
if anova_result.pvalue < alpha:
    print(f"Result: p-value ({anova_result.pvalue:.6f}) < {alpha}, reject H₀ → price differences are statistically significant.")
else:
    print(f"Result: p-value ({anova_result.pvalue:.6f}) ≥ {alpha}, fail to reject H₀ → price differences are not statistically significant.")

# Visualization 1: Average price across boroughs
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="neighbourhood_group", y="price", ci=None)
plt.title("Average Listing Price by Borough")
plt.xlabel("Borough")
plt.ylabel("Average Price ($)")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------
# 3. Correlation Between Number of Reviews and Price
# --------------------------------------------------------------------------------

# Measured the linear correlation between review count and price
corr = df["price"].corr(df["number_of_reviews"])
print(f"\n[3] Correlation between price and number_of_reviews: {corr:.4f}")

# Visualization 2: Scatter plot with regression line
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="number_of_reviews", y="price", alpha=0.3)
sns.regplot(data=df, x="number_of_reviews", y="price", scatter=False, color="red")
plt.title("Price vs Number of Reviews")
plt.xlabel("Number of Reviews")
plt.ylabel("Price ($)")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------
# 4. Availability by Room Type (ANOVA)
# --------------------------------------------------------------------------------

# Compared average yearly availability across room types
avg_avail = df.groupby("room_type")["availability_365"].mean()
print("\n[4] Average Availability by Room Type:\n", avg_avail)

# used ANOVA to test for significant availability differences
room_types = df["room_type"].unique()
avail_groups = [df[df["room_type"] == rt]["availability_365"] for rt in room_types]
avail_anova = f_oneway(*avail_groups)
print(f"ANOVA p-value: {avail_anova.pvalue:.6f}")

# Visualization 3: Violin plot to show distribution
plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x="room_type", y="availability_365")
plt.title("Availability by Room Type")
plt.xlabel("Room Type")
plt.ylabel("Availability (days/year)")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------
# 5. Price Variability by Borough
# --------------------------------------------------------------------------------

# Calculated standard deviation to see where prices vary the most
std_prices = df.groupby("neighbourhood_group")["price"].std()
print("\n[5] Price Standard Deviation by Borough:\n", std_prices)
print(f"Most price-variable borough: {std_prices.idxmax()}")

# Visualization 4: Box plot to visualize price spread
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="neighbourhood_group", y="price")
plt.title("Price Distribution by Borough")
plt.xlabel("Borough")
plt.ylabel("Price ($)")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------
# 6. Room Type vs Price
# --------------------------------------------------------------------------------

# Visualization 5: Box plot of prices across room types
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="room_type", y="price")
plt.title("Room Type vs Price")
plt.xlabel("Room Type")
plt.ylabel("Price ($)")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------
# 7. Outlier Identification: Price Histogram and Median Count
# --------------------------------------------------------------------------------

# price distribution with histogram and median line
plt.figure(figsize=(8, 5))
sns.histplot(df["price"], bins=50, kde=True)
plt.axvline(df["price"].median(), color='red', linestyle='--', label='Median')
plt.title("Price Distribution with Outliers")
plt.xlabel("Price ($)")
plt.legend()
plt.tight_layout()
plt.show()

# listings above/below the median to support analysis
median_price = df["price"].median()
above_median = (df["price"] > median_price).sum()
below_median = (df["price"] <= median_price).sum()
print(f"\n[7] Median Price: ${median_price:.2f}")
print(f"Listings above median: {above_median}")
print(f"Listings below or equal to median: {below_median}")

# --------------------------------------------------------------------------------
# 8. Correlation Heatmap
# --------------------------------------------------------------------------------

# Visualization 7: Heatmap for numeric features
plt.figure(figsize=(8, 6))
corr_matrix = df[["price", "number_of_reviews", "minimum_nights", "availability_365"]].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------
# 9. Linear Regression: Predicting Price from Reviews
# --------------------------------------------------------------------------------

# A simple linear regression model with number_of_reviews as predictor
X1 = df[["number_of_reviews"]]
y = df["price"]
model1 = LinearRegression().fit(X1, y)
print(f"\n[9] Linear Regression: number_of_reviews → price")
print(f"Slope: {model1.coef_[0]:.4f}, R2 Score: {model1.score(X1, y):.4f}")

# --------------------------------------------------------------------------------
# 10. Multiple Regression: Room Type and Availability
# --------------------------------------------------------------------------------

# Converted categorical room_type to numeric via one-hot encoding
df_enc = pd.get_dummies(df, columns=["room_type"], drop_first=True)
X2 = df_enc[["availability_365", "room_type_Private room", "room_type_Shared room"]]
model2 = LinearRegression().fit(X2, y)

print("\n[10] Multiple Regression: availability + room type → price")
print(f"R2 Score: {model2.score(X2, y):.4f}")
for feat, coef in zip(X2.columns, model2.coef_):
    print(f"{feat}: {coef:.4f}")

# --------------------------------------------------------------------------------
# 11. Hypothesis Testing: T-test Between Room Types
# --------------------------------------------------------------------------------

# Instead of checking one-hot encoded columns directly, filtering original df by room_type label
homes = df[df["room_type"] == "Entire home/apt"]["price"]
private = df[df["room_type"] == "Private room"]["price"]
t_stat, p_val = ttest_ind(homes, private, equal_var=False)

print("\n[11] T-test: Entire Home vs Private Room")
print(f"T-statistic: {t_stat:.4f}, p-value: {p_val:.6f}")
if p_val < 0.05:
    print("Conclusion: Entire homes cost significantly more than private rooms.")
else:
    print("Conclusion: No significant price difference between room types.")

# --------------------------------------------------------------------------------
# 12. Outlier Handling Recommendation
# --------------------------------------------------------------------------------

# Identified high-price and long minimum nights listings
price_outliers = df[df["price"] > 800]
min_night_outliers = df[df["minimum_nights"] >= 300]
print(f"\n[12] Outlier Detection")
print(f"\nListings with price > $800: {len(price_outliers)}")
print(f"Listings with minimum nights >= 300: {len(min_night_outliers)}")

# Created a new cleaned dataset with outliers removed for future analysis
df_cleaned = df[(df["price"] <= 800) & (df["minimum_nights"] < 300)]
print(f"Cleaned dataset size: {df_cleaned.shape[0]} listings (after removing outliers)")

# --------------------------------------------------------------------------------
# 13. Pearson Correlation Coefficient (Separate Section)
# --------------------------------------------------------------------------------

# This section measures the strength of the linear relationship between price and number of reviews.
pearson_coef, pearson_pval = pearsonr(df["price"], df["number_of_reviews"])
print("\n[13] Pearson Correlation Coefficient (Separate)")
print(f"Pearson Correlation Coefficient: {pearson_coef:.4f}")
print(f"P-value: {pearson_pval:.6f}")

# Interpretation
if abs(pearson_coef) < 0.3:
    strength = "a weak"
elif abs(pearson_coef) < 0.7:
    strength = "a moderate"
else:
    strength = "a strong"

print(f"This indicates {strength} linear relationship between price and number of reviews.")

# --------------------------------------------------------------------------------
# BUSINESS INSIGHTS SECTION
# --------------------------------------------------------------------------------

# Insight 1: Which borough generates the highest total revenue?
# Group by 'neighbourhood_group' and sum 'price'
# Helps hosts/investors target high-revenue areas.
rev_by_borough = df.groupby("neighbourhood_group")["price"].sum().sort_values(ascending=False)
print("\n[BI-1] Total Revenue by Borough:\n", rev_by_borough)
sns.barplot(x=rev_by_borough.index, y=rev_by_borough.values)
plt.title("Total Airbnb Revenue by Borough") 
plt.ylabel("Total Revenue ($)")
plt.show()


# Insight 2: What room types deliver the best revenue per availability day?
# Helps hosts understand yield per day serviced.
df["rev_per_day"] = df["price"] / (365 - df["availability_365"] + 1)
room_rev = df.groupby("room_type")["rev_per_day"].mean().sort_values(ascending=False)
print("\n[BI-3] Avg Revenue Per Day by Room Type:\n", room_rev)
sns.barplot(x=room_rev.index, y=room_rev.values)
plt.title("Avg Daily Revenue by Room Type")
plt.ylabel("Revenue per Available Day ($)")
plt.show()


# Insight 3: How does Airbnb supply concentration compare across boroughs?
# Reveals market saturation or opportunities in underrepresented areas :contentReference[oaicite:1]{index=1}.
# Calculate percentages of listings per borough
listings_per_borough = df["neighbourhood_group"].value_counts(normalize=True) * 100
print("\n[BI-5] % of Listings by Borough:\n", listings_per_borough)
colors = sns.color_palette('pastel', len(listings_per_borough))
fig, ax = plt.subplots(figsize=(8,8))

wedges, texts, autotexts = ax.pie(
    listings_per_borough.values,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    pctdistance=0.7
)

ax.axis('equal')

ax.legend(
    wedges,
    listings_per_borough.index,
    title="Boroughs",
    loc='center left',
    bbox_to_anchor=(1, 0.5)
)

plt.title("Airbnb Market Share by Borough")
plt.tight_layout()
plt.show()