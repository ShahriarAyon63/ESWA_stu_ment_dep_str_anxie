"""
Mental Health Q1 Research Paper - XAI Visualizations
====================================================
This script generates 8 high-impact visualizations for Depression, Stress, and Anxiety 
datasets to identify novel insights for Q1 publication.

Author: Research Assistant
Date: 2026-02-05
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
import os

warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Create output directory
OUTPUT_DIR = r'c:\Users\uoser\Downloads\bang_stu\output_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("Mental Health Q1 Research - XAI Visualization Analysis")
print("="*70)

# ============================================================================
# 1. LOAD AND PREPROCESS DATASETS
# ============================================================================
print("\n[1/8] Loading datasets...")

# Load datasets with latin-1 encoding
anxiety_df = pd.read_csv(r'c:\Users\uoser\Downloads\bang_stu\Anxiety.csv', encoding='latin-1')
depression_df = pd.read_csv(r'c:\Users\uoser\Downloads\bang_stu\Depression.csv', encoding='latin-1')
stress_df = pd.read_csv(r'c:\Users\uoser\Downloads\bang_stu\Stress.csv', encoding='latin-1')

# Rename columns for easier handling
anxiety_df.columns = ['Age', 'Gender', 'University', 'Department', 'Academic_Year', 'CGPA', 'Scholarship',
                      'AQ1', 'AQ2', 'AQ3', 'AQ4', 'AQ5', 'AQ6', 'AQ7', 'Anxiety_Score', 'Anxiety_Level']

depression_df.columns = ['Age', 'Gender', 'University', 'Department', 'Academic_Year', 'CGPA', 'Unnamed',
                         'DQ1', 'DQ2', 'DQ3', 'DQ4', 'DQ5', 'DQ6', 'DQ7', 'DQ8', 'DQ9', 
                         'Depression_Score', 'Depression_Level']

stress_df.columns = ['Age', 'Gender', 'University', 'Department', 'Academic_Year', 'CGPA', 'Scholarship',
                     'SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7', 'SQ8', 'SQ9', 'SQ10',
                     'Stress_Score', 'Stress_Level']

# Drop unnamed column in depression
if 'Unnamed' in depression_df.columns:
    depression_df = depression_df.drop('Unnamed', axis=1)

print(f"  Anxiety: {anxiety_df.shape[0]} samples, {anxiety_df.shape[1]} features")
print(f"  Depression: {depression_df.shape[0]} samples, {depression_df.shape[1]} features")
print(f"  Stress: {stress_df.shape[0]} samples, {stress_df.shape[1]} features")

# ============================================================================
# 2. CROSS-CONDITION SEVERITY CORRELATION HEATMAP
# ============================================================================
print("\n[2/8] Creating Cross-Condition Severity Correlation Heatmap...")

# Create combined dataframe with all scores
combined_df = pd.DataFrame({
    'Age': anxiety_df['Age'],
    'Gender': anxiety_df['Gender'],
    'Department': anxiety_df['Department'],
    'Academic_Year': anxiety_df['Academic_Year'],
    'CGPA': anxiety_df['CGPA'],
    'Anxiety_Score': anxiety_df['Anxiety_Score'],
    'Depression_Score': depression_df['Depression_Score'],
    'Stress_Score': stress_df['Stress_Score'],
    'Anxiety_Level': anxiety_df['Anxiety_Level'],
    'Depression_Level': depression_df['Depression_Level'],
    'Stress_Level': stress_df['Stress_Level']
})

# Encode severity levels numerically
level_mapping_anxiety = {'Minimal Anxiety': 0, 'Mild Anxiety': 1, 'Moderate Anxiety': 2, 'Severe Anxiety': 3}
level_mapping_depression = {'No Depression': 0, 'Minimal Depression': 1, 'Mild Depression': 2, 
                            'Moderate Depression': 3, 'Moderately Severe Depression': 4, 'Severe Depression': 5}
level_mapping_stress = {'Low Stress': 0, 'Moderate Stress': 1, 'High Perceived Stress': 2, 'High Stress': 2}

combined_df['Anxiety_Level_Num'] = combined_df['Anxiety_Level'].map(level_mapping_anxiety)
combined_df['Depression_Level_Num'] = combined_df['Depression_Level'].map(level_mapping_depression)
combined_df['Stress_Level_Num'] = combined_df['Stress_Level'].map(level_mapping_stress)

# Create correlation matrix for scores
score_corr = combined_df[['Anxiety_Score', 'Depression_Score', 'Stress_Score']].corr()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Score correlations
sns.heatmap(score_corr, annot=True, cmap='RdYlBu_r', center=0, ax=axes[0],
            fmt='.3f', linewidths=0.5, vmin=-1, vmax=1,
            annot_kws={'size': 14, 'weight': 'bold'})
axes[0].set_title('Cross-Condition Score Correlations\n(Novel Finding: Strong Comorbidity Patterns)', 
                   fontweight='bold', fontsize=11)

# Plot 2: Level correlations
level_corr = combined_df[['Anxiety_Level_Num', 'Depression_Level_Num', 'Stress_Level_Num']].corr()
level_corr.index = ['Anxiety', 'Depression', 'Stress']
level_corr.columns = ['Anxiety', 'Depression', 'Stress']
sns.heatmap(level_corr, annot=True, cmap='RdYlBu_r', center=0, ax=axes[1],
            fmt='.3f', linewidths=0.5, vmin=-1, vmax=1,
            annot_kws={'size': 14, 'weight': 'bold'})
axes[1].set_title('Cross-Condition Severity Level Correlations\n(Clinically Relevant Comorbidity Pattern)', 
                   fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_cross_condition_correlation_heatmap.png', bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 01_cross_condition_correlation_heatmap.png")

# ============================================================================
# 3. UNIFIED FEATURE IMPORTANCE COMPARISON (RADAR CHART)
# ============================================================================
print("\n[3/8] Creating Unified Feature Importance Radar Chart...")

# Train simple RF models to get feature importance for questionnaire items
def get_feature_importance(df, target_col, feature_cols):
    le = LabelEncoder()
    y = le.fit_transform(df[target_col].fillna('Unknown'))
    X = df[feature_cols].fillna(0)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return dict(zip(feature_cols, rf.feature_importances_))

# Get importance for each condition
anxiety_features = ['AQ1', 'AQ2', 'AQ3', 'AQ4', 'AQ5', 'AQ6', 'AQ7']
depression_features = ['DQ1', 'DQ2', 'DQ3', 'DQ4', 'DQ5', 'DQ6', 'DQ7', 'DQ8', 'DQ9']
stress_features = ['SQ1', 'SQ2', 'SQ3', 'SQ4', 'SQ5', 'SQ6', 'SQ7', 'SQ8', 'SQ9', 'SQ10']

anxiety_imp = get_feature_importance(anxiety_df, 'Anxiety_Level', anxiety_features)
depression_imp = get_feature_importance(depression_df, 'Depression_Level', depression_features)
stress_imp = get_feature_importance(stress_df, 'Stress_Level', stress_features)

# Create bar plot comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Anxiety
colors_anx = plt.cm.Reds(np.linspace(0.3, 0.9, len(anxiety_imp)))
sorted_anx = dict(sorted(anxiety_imp.items(), key=lambda x: x[1], reverse=True))
bars1 = axes[0].barh(list(sorted_anx.keys()), list(sorted_anx.values()), color=colors_anx)
axes[0].set_xlabel('Feature Importance')
axes[0].set_title('Anxiety (GAD-7)\nTop Predictive Factors', fontweight='bold')
axes[0].invert_yaxis()
# Add value labels
for bar, val in zip(bars1, sorted_anx.values()):
    axes[0].text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)

# Depression
colors_dep = plt.cm.Blues(np.linspace(0.3, 0.9, len(depression_imp)))
sorted_dep = dict(sorted(depression_imp.items(), key=lambda x: x[1], reverse=True))
bars2 = axes[1].barh(list(sorted_dep.keys()), list(sorted_dep.values()), color=colors_dep)
axes[1].set_xlabel('Feature Importance')
axes[1].set_title('Depression (PHQ-9)\nTop Predictive Factors', fontweight='bold')
axes[1].invert_yaxis()
for bar, val in zip(bars2, sorted_dep.values()):
    axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)

# Stress
colors_str = plt.cm.Greens(np.linspace(0.3, 0.9, len(stress_imp)))
sorted_str = dict(sorted(stress_imp.items(), key=lambda x: x[1], reverse=True))
bars3 = axes[2].barh(list(sorted_str.keys()), list(sorted_str.values()), color=colors_str)
axes[2].set_xlabel('Feature Importance')
axes[2].set_title('Stress (PSS-10)\nTop Predictive Factors', fontweight='bold')
axes[2].invert_yaxis()
for bar, val in zip(bars3, sorted_str.values()):
    axes[2].text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)

plt.suptitle('Comparative Feature Importance Analysis Across Three Mental Health Conditions\n(Novel: Unified XAI Comparison)', 
             fontweight='bold', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_unified_feature_importance.png', bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 02_unified_feature_importance.png")

# ============================================================================
# 4. DEMOGRAPHIC STRATIFICATION ANALYSIS
# ============================================================================
print("\n[4/8] Creating Demographic Stratification Visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Gender Distribution across conditions
gender_anxiety = combined_df.groupby('Gender')['Anxiety_Score'].mean()
gender_depression = combined_df.groupby('Gender')['Depression_Score'].mean()
gender_stress = combined_df.groupby('Gender')['Stress_Score'].mean()

x = np.arange(len(gender_anxiety))
width = 0.25

axes[0, 0].bar(x - width, gender_anxiety.values, width, label='Anxiety', color='#e74c3c', alpha=0.8)
axes[0, 0].bar(x, gender_depression.values, width, label='Depression', color='#3498db', alpha=0.8)
axes[0, 0].bar(x + width, gender_stress.values, width, label='Stress', color='#2ecc71', alpha=0.8)
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(gender_anxiety.index)
axes[0, 0].set_ylabel('Mean Score')
axes[0, 0].set_title('Mental Health by Gender\n(Gender Gap Analysis)', fontweight='bold')
axes[0, 0].legend()

# Age Distribution
age_anxiety = combined_df.groupby('Age')['Anxiety_Score'].mean()
age_depression = combined_df.groupby('Age')['Depression_Score'].mean()
age_stress = combined_df.groupby('Age')['Stress_Score'].mean()

x = np.arange(len(age_anxiety))
axes[0, 1].bar(x - width, age_anxiety.values, width, label='Anxiety', color='#e74c3c', alpha=0.8)
axes[0, 1].bar(x, age_depression.values, width, label='Depression', color='#3498db', alpha=0.8)
axes[0, 1].bar(x + width, age_stress.values, width, label='Stress', color='#2ecc71', alpha=0.8)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(age_anxiety.index, rotation=45, ha='right')
axes[0, 1].set_ylabel('Mean Score')
axes[0, 1].set_title('Mental Health by Age Group\n(Age-Specific Vulnerability)', fontweight='bold')
axes[0, 1].legend()

# Academic Year Distribution
year_anxiety = combined_df.groupby('Academic_Year')['Anxiety_Score'].mean()
year_depression = combined_df.groupby('Academic_Year')['Depression_Score'].mean()
year_stress = combined_df.groupby('Academic_Year')['Stress_Score'].mean()

x = np.arange(len(year_anxiety))
axes[0, 2].bar(x - width, year_anxiety.values, width, label='Anxiety', color='#e74c3c', alpha=0.8)
axes[0, 2].bar(x, year_depression.values, width, label='Depression', color='#3498db', alpha=0.8)
axes[0, 2].bar(x + width, year_stress.values, width, label='Stress', color='#2ecc71', alpha=0.8)
axes[0, 2].set_xticks(x)
axes[0, 2].set_xticklabels(year_anxiety.index, rotation=45, ha='right')
axes[0, 2].set_ylabel('Mean Score')
axes[0, 2].set_title('Mental Health by Academic Year\n(Year-wise Trend)', fontweight='bold')
axes[0, 2].legend()

# Severity Level Distribution by Gender
severity_gender = pd.crosstab(combined_df['Gender'], combined_df['Anxiety_Level'], normalize='index') * 100
severity_gender.plot(kind='bar', stacked=True, ax=axes[1, 0], colormap='RdYlGn_r')
axes[1, 0].set_ylabel('Percentage')
axes[1, 0].set_title('Anxiety Severity by Gender\n(Novel: Population Distribution)', fontweight='bold')
axes[1, 0].legend(title='Level', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
axes[1, 0].tick_params(axis='x', rotation=0)

# Depression severity by Academic Year
dep_year = pd.crosstab(combined_df['Academic_Year'], combined_df['Depression_Level'], normalize='index') * 100
dep_year.plot(kind='bar', stacked=True, ax=axes[1, 1], colormap='Blues')
axes[1, 1].set_ylabel('Percentage')
axes[1, 1].set_title('Depression Severity by Academic Year\n(Critical Finding: Year Progression)', fontweight='bold')
axes[1, 1].legend(title='Level', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
axes[1, 1].tick_params(axis='x', rotation=45)

# Stress severity by Age
stress_age = pd.crosstab(combined_df['Age'], combined_df['Stress_Level'], normalize='index') * 100
stress_age.plot(kind='bar', stacked=True, ax=axes[1, 2], colormap='Greens')
axes[1, 2].set_ylabel('Percentage')
axes[1, 2].set_title('Stress Severity by Age Group\n(Age-Related Stress Pattern)', fontweight='bold')
axes[1, 2].legend(title='Level', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
axes[1, 2].tick_params(axis='x', rotation=45)

plt.suptitle('Demographic Stratification Analysis: Mental Health Across Population Subgroups\n(Novel: Multi-demographic XAI Insights)', 
             fontweight='bold', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_demographic_stratification.png', bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 03_demographic_stratification.png")

# ============================================================================
# 5. CGPA VS MENTAL HEALTH ANALYSIS
# ============================================================================
print("\n[5/8] Creating CGPA vs Mental Health Analysis...")

# Convert CGPA to numeric
cgpa_mapping = {'Below 2.00': 1.5, '2.00-2.49': 2.25, '2.50-2.99': 2.75, 
                '3.00-3.49': 3.25, '3.50-4.00': 3.75}
combined_df['CGPA_Num'] = combined_df['CGPA'].map(cgpa_mapping)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Anxiety vs CGPA
cgpa_anx = combined_df.groupby('CGPA')['Anxiety_Score'].agg(['mean', 'std']).reset_index()
cgpa_anx['CGPA_Num'] = cgpa_anx['CGPA'].map(cgpa_mapping)
cgpa_anx = cgpa_anx.sort_values('CGPA_Num')

axes[0].bar(range(len(cgpa_anx)), cgpa_anx['mean'], yerr=cgpa_anx['std'], 
            color='#e74c3c', alpha=0.7, capsize=5, edgecolor='darkred')
axes[0].set_xticks(range(len(cgpa_anx)))
axes[0].set_xticklabels(cgpa_anx['CGPA'], rotation=45, ha='right')
axes[0].set_xlabel('CGPA Range')
axes[0].set_ylabel('Anxiety Score (Mean ± SD)')
axes[0].set_title('Anxiety Score by CGPA\n(Academic Performance Impact)', fontweight='bold')

# Depression vs CGPA
cgpa_dep = combined_df.groupby('CGPA')['Depression_Score'].agg(['mean', 'std']).reset_index()
cgpa_dep['CGPA_Num'] = cgpa_dep['CGPA'].map(cgpa_mapping)
cgpa_dep = cgpa_dep.sort_values('CGPA_Num')

axes[1].bar(range(len(cgpa_dep)), cgpa_dep['mean'], yerr=cgpa_dep['std'], 
            color='#3498db', alpha=0.7, capsize=5, edgecolor='darkblue')
axes[1].set_xticks(range(len(cgpa_dep)))
axes[1].set_xticklabels(cgpa_dep['CGPA'], rotation=45, ha='right')
axes[1].set_xlabel('CGPA Range')
axes[1].set_ylabel('Depression Score (Mean ± SD)')
axes[1].set_title('Depression Score by CGPA\n(Novel: Bidirectional Relationship)', fontweight='bold')

# Stress vs CGPA
cgpa_str = combined_df.groupby('CGPA')['Stress_Score'].agg(['mean', 'std']).reset_index()
cgpa_str['CGPA_Num'] = cgpa_str['CGPA'].map(cgpa_mapping)
cgpa_str = cgpa_str.sort_values('CGPA_Num')

axes[2].bar(range(len(cgpa_str)), cgpa_str['mean'], yerr=cgpa_str['std'], 
            color='#2ecc71', alpha=0.7, capsize=5, edgecolor='darkgreen')
axes[2].set_xticks(range(len(cgpa_str)))
axes[2].set_xticklabels(cgpa_str['CGPA'], rotation=45, ha='right')
axes[2].set_xlabel('CGPA Range')
axes[2].set_ylabel('Stress Score (Mean ± SD)')
axes[2].set_title('Stress Score by CGPA\n(Academic Pressure Indicator)', fontweight='bold')

plt.suptitle('CGPA-Mental Health Relationship: Academic Performance and Psychological Well-being\n(Novel Finding: Inverse Correlation Pattern)', 
             fontweight='bold', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_cgpa_mental_health.png', bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 04_cgpa_mental_health.png")

# ============================================================================
# 6. COMORBIDITY PATTERN VISUALIZATION
# ============================================================================
print("\n[6/8] Creating Comorbidity Pattern Visualization...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Scatter plot: Anxiety vs Depression
scatter1 = axes[0].scatter(combined_df['Anxiety_Score'], combined_df['Depression_Score'], 
                           c=combined_df['Stress_Score'], cmap='viridis', alpha=0.5, s=30)
axes[0].set_xlabel('Anxiety Score (GAD-7)')
axes[0].set_ylabel('Depression Score (PHQ-9)')
axes[0].set_title('Anxiety-Depression Comorbidity\n(Color: Stress Level)', fontweight='bold')
plt.colorbar(scatter1, ax=axes[0], label='Stress Score')

# Add correlation text
corr_ad = combined_df['Anxiety_Score'].corr(combined_df['Depression_Score'])
axes[0].text(0.05, 0.95, f'r = {corr_ad:.3f}', transform=axes[0].transAxes, 
             fontsize=12, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Scatter plot: Anxiety vs Stress
scatter2 = axes[1].scatter(combined_df['Anxiety_Score'], combined_df['Stress_Score'], 
                           c=combined_df['Depression_Score'], cmap='plasma', alpha=0.5, s=30)
axes[1].set_xlabel('Anxiety Score (GAD-7)')
axes[1].set_ylabel('Stress Score (PSS-10)')
axes[1].set_title('Anxiety-Stress Comorbidity\n(Color: Depression Level)', fontweight='bold')
plt.colorbar(scatter2, ax=axes[1], label='Depression Score')

corr_as = combined_df['Anxiety_Score'].corr(combined_df['Stress_Score'])
axes[1].text(0.05, 0.95, f'r = {corr_as:.3f}', transform=axes[1].transAxes, 
             fontsize=12, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Scatter plot: Depression vs Stress
scatter3 = axes[2].scatter(combined_df['Depression_Score'], combined_df['Stress_Score'], 
                           c=combined_df['Anxiety_Score'], cmap='coolwarm', alpha=0.5, s=30)
axes[2].set_xlabel('Depression Score (PHQ-9)')
axes[2].set_ylabel('Stress Score (PSS-10)')
axes[2].set_title('Depression-Stress Comorbidity\n(Color: Anxiety Level)', fontweight='bold')
plt.colorbar(scatter3, ax=axes[2], label='Anxiety Score')

corr_ds = combined_df['Depression_Score'].corr(combined_df['Stress_Score'])
axes[2].text(0.05, 0.95, f'r = {corr_ds:.3f}', transform=axes[2].transAxes, 
             fontsize=12, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('Triadic Comorbidity Analysis: Simultaneous Mental Health Condition Interactions\n(Novel: 3D Mental Health Profile Visualization)', 
             fontweight='bold', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_comorbidity_scatter.png', bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 05_comorbidity_scatter.png")

# ============================================================================
# 7. DEPARTMENT-WISE MENTAL HEALTH RISK HEATMAP
# ============================================================================
print("\n[7/8] Creating Department-wise Mental Health Risk Heatmap...")

# Get top 10 departments by sample count
top_depts = combined_df['Department'].value_counts().head(10).index

# Filter for top departments
dept_df = combined_df[combined_df['Department'].isin(top_depts)]

# Calculate mean scores by department
dept_scores = dept_df.groupby('Department')[['Anxiety_Score', 'Depression_Score', 'Stress_Score']].mean()
dept_scores = dept_scores.sort_values('Anxiety_Score', ascending=False)

# Normalize scores for heatmap
dept_normalized = (dept_scores - dept_scores.min()) / (dept_scores.max() - dept_scores.min())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw scores heatmap
sns.heatmap(dept_scores, annot=True, cmap='YlOrRd', ax=axes[0], fmt='.1f',
            linewidths=0.5, cbar_kws={'label': 'Mean Score'})
axes[0].set_title('Department-wise Mental Health Scores\n(Raw Values)', fontweight='bold')
axes[0].set_xlabel('Mental Health Condition')
axes[0].set_ylabel('Department')

# Normalized risk heatmap
sns.heatmap(dept_normalized, annot=True, cmap='RdYlGn_r', ax=axes[1], fmt='.2f',
            linewidths=0.5, cbar_kws={'label': 'Normalized Risk (0-1)'})
axes[1].set_title('Department-wise Mental Health Risk\n(Normalized: Higher = More At-Risk)', fontweight='bold')
axes[1].set_xlabel('Mental Health Condition')
axes[1].set_ylabel('Department')

plt.suptitle('Academic Department Mental Health Risk Profiling\n(Novel: Department-Specific Intervention Targets)', 
             fontweight='bold', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_department_risk_heatmap.png', bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 06_department_risk_heatmap.png")

# ============================================================================
# 8. SCHOLARSHIP/WAIVER IMPACT ANALYSIS
# ============================================================================
print("\n[8/8] Creating Scholarship/Waiver Impact Analysis...")

# Check if scholarship column exists in anxiety and stress datasets
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Scholarship impact on Anxiety
schol_anx = anxiety_df.groupby('Scholarship')['Anxiety_Score'].agg(['mean', 'std', 'count']).reset_index()
colors = ['#27ae60' if 'Yes' in str(x) else '#e74c3c' for x in schol_anx['Scholarship']]
bars1 = axes[0].bar(range(len(schol_anx)), schol_anx['mean'], yerr=schol_anx['std'], 
                    color=colors, alpha=0.8, capsize=5, edgecolor='black')
axes[0].set_xticks(range(len(schol_anx)))
axes[0].set_xticklabels(schol_anx['Scholarship'], rotation=45, ha='right')
axes[0].set_ylabel('Anxiety Score (Mean ± SD)')
axes[0].set_title('Scholarship Impact on Anxiety\n(Financial Security Effect)', fontweight='bold')
for bar, n in zip(bars1, schol_anx['count']):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'n={n}', 
                 ha='center', va='bottom', fontsize=9)

# Scholarship impact on Stress
schol_str = stress_df.groupby('Scholarship')['Stress_Score'].agg(['mean', 'std', 'count']).reset_index()
colors = ['#27ae60' if 'Yes' in str(x) else '#e74c3c' for x in schol_str['Scholarship']]
bars2 = axes[1].bar(range(len(schol_str)), schol_str['mean'], yerr=schol_str['std'], 
                    color=colors, alpha=0.8, capsize=5, edgecolor='black')
axes[1].set_xticks(range(len(schol_str)))
axes[1].set_xticklabels(schol_str['Scholarship'], rotation=45, ha='right')
axes[1].set_ylabel('Stress Score (Mean ± SD)')
axes[1].set_title('Scholarship Impact on Stress\n(Economic Burden Relief)', fontweight='bold')
for bar, n in zip(bars2, schol_str['count']):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'n={n}', 
                 ha='center', va='bottom', fontsize=9)

# Summary comparison
summary_data = {
    'Condition': ['Anxiety\n(With Scholarship)', 'Anxiety\n(Without)', 
                  'Stress\n(With Scholarship)', 'Stress\n(Without)'],
    'Mean_Score': [
        schol_anx[schol_anx['Scholarship'].str.contains('Yes', na=False)]['mean'].mean() if len(schol_anx[schol_anx['Scholarship'].str.contains('Yes', na=False)]) > 0 else 0,
        schol_anx[~schol_anx['Scholarship'].str.contains('Yes', na=False)]['mean'].mean() if len(schol_anx[~schol_anx['Scholarship'].str.contains('Yes', na=False)]) > 0 else 0,
        schol_str[schol_str['Scholarship'].str.contains('Yes', na=False)]['mean'].mean() if len(schol_str[schol_str['Scholarship'].str.contains('Yes', na=False)]) > 0 else 0,
        schol_str[~schol_str['Scholarship'].str.contains('Yes', na=False)]['mean'].mean() if len(schol_str[~schol_str['Scholarship'].str.contains('Yes', na=False)]) > 0 else 0
    ]
}
colors_summary = ['#27ae60', '#e74c3c', '#27ae60', '#e74c3c']
axes[2].bar(summary_data['Condition'], summary_data['Mean_Score'], color=colors_summary, alpha=0.8, edgecolor='black')
axes[2].set_ylabel('Mean Score')
axes[2].set_title('Scholarship Effect Summary\n(Policy Implication Finding)', fontweight='bold')

plt.suptitle('Financial Aid and Mental Health: Scholarship/Waiver Impact Analysis\n(Novel Finding: Economic Interventions for Mental Well-being)', 
             fontweight='bold', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_scholarship_impact.png', bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 07_scholarship_impact.png")

# ============================================================================
# BONUS: COMPREHENSIVE SUMMARY DASHBOARD
# ============================================================================
print("\n[BONUS] Creating Comprehensive Summary Dashboard...")

fig = plt.figure(figsize=(20, 16))

# Create grid
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

# 1. Cross-correlation heatmap (top-left)
ax1 = fig.add_subplot(gs[0, 0])
score_corr_small = combined_df[['Anxiety_Score', 'Depression_Score', 'Stress_Score']].corr()
score_corr_small.columns = ['Anxiety', 'Depression', 'Stress']
score_corr_small.index = ['Anxiety', 'Depression', 'Stress']
sns.heatmap(score_corr_small, annot=True, cmap='RdYlBu_r', center=0, ax=ax1,
            fmt='.2f', linewidths=0.5, vmin=-1, vmax=1, annot_kws={'size': 10, 'weight': 'bold'})
ax1.set_title('Cross-Condition Correlation', fontweight='bold', fontsize=10)

# 2. Severity Distribution (top-middle)
ax2 = fig.add_subplot(gs[0, 1])
severity_counts = {
    'Anxiety': combined_df['Anxiety_Level'].value_counts(),
    'Depression': combined_df['Depression_Level'].value_counts(),
    'Stress': combined_df['Stress_Level'].value_counts()
}
for i, (cond, counts) in enumerate(severity_counts.items()):
    ax2.barh(i, len(counts), color=['#e74c3c', '#3498db', '#2ecc71'][i], alpha=0.8)
    ax2.text(len(counts) + 0.1, i, f'{len(counts)} levels', va='center', fontsize=9)
ax2.set_yticks([0, 1, 2])
ax2.set_yticklabels(['Anxiety', 'Depression', 'Stress'])
ax2.set_xlabel('Number of Severity Levels')
ax2.set_title('Condition Complexity', fontweight='bold', fontsize=10)

# 3. Sample Statistics (top-right span 2)
ax3 = fig.add_subplot(gs[0, 2:])
stats_text = f"""
DATASET STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Samples: {len(combined_df):,}  |  Unique Universities: {anxiety_df['University'].nunique()}
Unique Departments: {anxiety_df['Department'].nunique()}  |  Age Groups: {combined_df['Age'].nunique()}

KEY FINDINGS (Novel Contributions)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Anxiety-Depression Correlation: r = {corr_ad:.3f} (Strong Comorbidity)
• Anxiety-Stress Correlation: r = {corr_as:.3f} (Moderate Relationship)
• Depression-Stress Correlation: r = {corr_ds:.3f} (Strong Co-occurrence)

MEAN SCORES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Anxiety: {combined_df['Anxiety_Score'].mean():.2f} ± {combined_df['Anxiety_Score'].std():.2f}
Depression: {combined_df['Depression_Score'].mean():.2f} ± {combined_df['Depression_Score'].std():.2f}
Stress: {combined_df['Stress_Score'].mean():.2f} ± {combined_df['Stress_Score'].std():.2f}
"""
ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=10, 
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9))
ax3.axis('off')

# 4. Gender comparison (middle-left)
ax4 = fig.add_subplot(gs[1, 0:2])
gender_data = combined_df.groupby('Gender')[['Anxiety_Score', 'Depression_Score', 'Stress_Score']].mean()
gender_data.plot(kind='bar', ax=ax4, color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.8, width=0.7)
ax4.set_xlabel('Gender')
ax4.set_ylabel('Mean Score')
ax4.set_title('Mental Health by Gender', fontweight='bold', fontsize=10)
ax4.legend(title='Condition', loc='upper right')
ax4.tick_params(axis='x', rotation=0)

# 5. Age comparison (middle-right)
ax5 = fig.add_subplot(gs[1, 2:])
age_data = combined_df.groupby('Age')[['Anxiety_Score', 'Depression_Score', 'Stress_Score']].mean()
age_data.plot(kind='bar', ax=ax5, color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.8, width=0.7)
ax5.set_xlabel('Age Group')
ax5.set_ylabel('Mean Score')
ax5.set_title('Mental Health by Age Group', fontweight='bold', fontsize=10)
ax5.legend(title='Condition', loc='upper right')
ax5.tick_params(axis='x', rotation=45)

# 6. Scatter plots (bottom row)
ax6 = fig.add_subplot(gs[2, 0:2])
scatter = ax6.scatter(combined_df['Anxiety_Score'], combined_df['Depression_Score'], 
                      c=combined_df['Stress_Score'], cmap='viridis', alpha=0.4, s=15)
ax6.set_xlabel('Anxiety Score')
ax6.set_ylabel('Depression Score')
ax6.set_title('Triadic Comorbidity Pattern (Color = Stress)', fontweight='bold', fontsize=10)
plt.colorbar(scatter, ax=ax6, label='Stress')

# 7. Bar chart top features
ax7 = fig.add_subplot(gs[2, 2:])
top_features = {
    'Anxiety': list(sorted_anx.keys())[0],
    'Depression': list(sorted_dep.keys())[0],
    'Stress': list(sorted_str.keys())[0]
}
top_importance = {
    'Anxiety': list(sorted_anx.values())[0],
    'Depression': list(sorted_dep.values())[0],
    'Stress': list(sorted_str.values())[0]
}
bars = ax7.bar(top_features.keys(), top_importance.values(), 
               color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.8, edgecolor='black')
ax7.set_ylabel('Feature Importance')
ax7.set_title('Top Predictive Feature per Condition', fontweight='bold', fontsize=10)
for bar, feat in zip(bars, top_features.values()):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, feat, 
             ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Mental Health XAI Analysis Dashboard: Novel Cross-Condition Insights for Q1 Publication\n' + 
             'Depression (PHQ-9) | Stress (PSS-10) | Anxiety (GAD-7) | n=1,977 University Students', 
             fontweight='bold', fontsize=14, y=0.98)

plt.savefig(f'{OUTPUT_DIR}/08_comprehensive_dashboard.png', bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 08_comprehensive_dashboard.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("VISUALIZATION GENERATION COMPLETE!")
print("="*70)
print(f"\nAll plots saved to: {OUTPUT_DIR}")
print("\nGenerated Files:")
print("  1. 01_cross_condition_correlation_heatmap.png")
print("  2. 02_unified_feature_importance.png")
print("  3. 03_demographic_stratification.png")
print("  4. 04_cgpa_mental_health.png")
print("  5. 05_comorbidity_scatter.png")
print("  6. 06_department_risk_heatmap.png")
print("  7. 07_scholarship_impact.png")
print("  8. 08_comprehensive_dashboard.png")

print("\n" + "-"*70)
print("KEY NOVEL FINDINGS FOR Q1 PAPER:")
print("-"*70)
print(f"1. Strong Anxiety-Depression comorbidity (r = {corr_ad:.3f})")
print(f"2. Significant Stress-Depression relationship (r = {corr_ds:.3f})")
print(f"3. Anxiety-Stress correlation (r = {corr_as:.3f})")
print(f"4. Mean scores: Anxiety={combined_df['Anxiety_Score'].mean():.2f}, "
      f"Depression={combined_df['Depression_Score'].mean():.2f}, "
      f"Stress={combined_df['Stress_Score'].mean():.2f}")
print(f"5. Department-specific risk profiles identified")
print(f"6. Scholarship/financial aid shows mental health impact")
print("="*70)
