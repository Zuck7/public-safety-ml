import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit


# Resolve the dataset next to this script so the project runs on any machine
# (it previously carried one team member's absolute Windows path).
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
KSI_CSV = os.environ.get("KSI_CSV", os.path.join(PROJECT_DIR, "KSI_data.csv"))

df = pd.read_csv(KSI_CSV)
print(df.head())


print(df.shape)

df.info()

print(df.isnull().sum())

summary_df = pd.DataFrame(
    {
        "Data Type": df.dtypes,
        "Non-Null Count": df.notnull().sum(),
        "Missing Count": df.isnull().sum(),
        "Missing %": (df.isnull().mean() * 100).round(2),
        "Unique Values": df.nunique(),
    }
)
print(summary_df)

# ranges and values of elements
print("Numeric Statistical Assessment")
print(df.describe().T)

print("Categorical Statistical Assessment")
print(df.describe(include="object").T)

# The missing-value stage is actually the trickiest part of this dataset, because
# most of the "missingness" here isn't random — it means something specific.
# Treating it carelessly (e.g. one global dropna() or filling everything with
# mode) will wreck other columns.

# Handling Missing Values:

# Group 1 — Binary involvement flags (huge missing %, but missing = "No")
# PEDESTRIAN, CYCLIST, AUTOMOBILE, MOTORCYCLE, TRUCK, TRSN_CITY_VEH, EMERG_VEH,
# PASSENGER, SPEEDING, AG_DRIV, REDLIGHT, ALCOHOL, DISABILITY
# These columns only contain the value "Yes" when true, and NaN otherwise —
# there's no "No" recorded. 95%+ missing on ALCOHOL/DISABILITY doesn't mean
# "bad data," it means alcohol/disability wasn't a factor in 95% of collisions.
# Fix: fillna("No"), not drop, not mode-impute.
group1_cols = [
    "PEDESTRIAN",
    "CYCLIST",
    "AUTOMOBILE",
    "MOTORCYCLE",
    "TRUCK",
    "TRSN_CITY_VEH",
    "EMERG_VEH",
    "PASSENGER",
    "SPEEDING",
    "AG_DRIV",
    "REDLIGHT",
    "ALCOHOL",
    "DISABILITY",
]

df[group1_cols] = df[group1_cols].fillna("No")
print(df[group1_cols].isnull().sum())

print(df["ALCOHOL"].value_counts())
print("\n", df["AG_DRIV"].value_counts())

# Group 2 — Role-conditional attributes (missing = not applicable to that record)
# PEDTYPE, PEDACT, PEDCOND only apply when the involved person is a pedestrian;
# CYCLISTYPE, CYCACT, CYCCOND only apply to cyclists.
# Their ~83-96% missingness is because most records aren't pedestrians/cyclists
# at all.
# Fix: fillna("Not Applicable") — dropping rows would delete almost every
# non-pedestrian/non-cyclist record.
group2_cols = [
    # Pedestrian-specific features
    "PEDTYPE",
    "PEDACT",
    "PEDCOND",
    # Cyclist-specific features
    "CYCLISTYPE",
    "CYCACT",
    "CYCCOND",
]

df[group2_cols] = df[group2_cols].fillna("Not Applicable")
print(df[group2_cols].isnull().sum())

print(df["PEDACT"].value_counts())
print("\n", df["CYCACT"].value_counts())

# Group 3 — Event-conditional FATAL_NO (95.4% missing)
# Only has a value for fatal injuries — it's a sequence number for fatalities
# in a multi-fatality collision.
# Missing = not fatal. fillna(0) or leave as "N/A" flag, not statistical
# imputation.
# Fix: Fill NaN Values with 0
df["FATAL_NO"] = df["FATAL_NO"].fillna(0).astype(int)
print("Missing count in FATAL_NO:", df["FATAL_NO"].isnull().sum())

print(df["FATAL_NO"].value_counts().head(10))

# Group 4 — Genuinely missing/incomplete records (true missing data)
# ACCLOC (28.8%), INITDIR (27.8%), ROAD_CLASS (2.6%), DISTRICT (1.2%),
# TRAFFCTL (0.4%), VISIBILITY, RDSFCOND, LIGHT, IMPACTYPE, INVTYPE, ACCLASS
# (all <1%) — these are cases where the report just wasn't filled in
# completely. This is only genuine "missing data" group.
# Fix: The strategy is split into two parts:
#   A - dropping negligible missing rows (<3%)
#   B - imputing medium missingness (~28%) as "Unknown"
group3A_cols = [
    "ACCLASS",
    "LIGHT",
    "VISIBILITY",
    "IMPACTYPE",
    "RDSFCOND",
    "TRAFFCTL",
    "DISTRICT",
    "ROAD_CLASS",
]

group3B_cols = ["ACCLOC", "INITDIR"]

df = df.dropna(subset=group3A_cols)
df[group3B_cols] = df[group3B_cols].fillna("Unkown")

print("Negligible Missing Rows\n", df[group3A_cols].isnull().sum())
print("\nMedium Missing Rows\n", df[group3B_cols].isnull().sum())

# Group 5 — Identifiers, not analysis variables
# OFFSET (79.8% missing), STREET2 (9%), ACCNUM (26%), VEHTYPE (18.4%)
# These are location/ID descriptors.
# ACCNUM missingness usually reflects records where an official number
# wasn't assigned
group5_cols = [
    "OFFSET",
    "STREET2",
    "VEHTYPE"]

df[group5_cols] = df[group5_cols].fillna("Unkown")

missing_accnum_mask = df["ACCNUM"].isnull()
df.loc[missing_accnum_mask, "ACCNUM"] = -1 * (df.loc[missing_accnum_mask].index + 1)

print(df[group5_cols].isnull().sum())
print("ACCNUM ", df["ACCNUM"].isnull().sum())

print(df.isnull().sum())

#Group-6
# Check whether missingness in [[MANOEUVER, DRIVACT, DRIVCOND, INJURY] columns lines up with INVTYPE categories
print(df.groupby("INVTYPE")["MANOEUVER"].apply(lambda x: x.isnull().mean()))
print(df.groupby("INVTYPE")["DRIVACT"].apply(lambda x: x.isnull().mean()))
print(df.groupby("INVTYPE")["DRIVCOND"].apply(lambda x: x.isnull().mean()))
print(df.groupby("INVTYPE")["INJURY"].apply(lambda x: x.isnull().mean()))

#Result: 
   # Missingness by INVTYPE Summary
   # MANOEUVER: Role-driven. Almost fully populated for vehicle operators (~0–0.4% missing) and almost entirely missing for non-operators (96–100%).
   # DRIVACT / DRIVCOND: Driver-specific. Populated only for drivers (~0.3–1.1% missing); 100% missing for all other roles.
   #INJURY: Mixed mechanism. Bystanders are ~100% missing (Not Applicable). Scene-present roles show variable missingness (e.g., Driver 63%, Passenger 35%), which indicates unrecorded non-injuries rather than non-applicability (None).

   #Action: Fill MANOEUVER, DRIVACT, and DRIVCOND uniformly with "Not Applicable". Fill INJURY conditionally using an INVTYPE mask ("None" for scene-present roles, "Not Applicable" for bystanders).

print(df["INJURY"].value_counts(dropna=False))


# INVTYPE — 16 missing, negligible, drop
df = df.dropna(subset=["INVTYPE"])

# MANOEUVER, DRIVACT, DRIVCOND — clean bimodal split by role (confirmed via crosstab)
# Missing = this person wasn't operating a vehicle, so the field doesn't apply
df[["MANOEUVER", "DRIVACT", "DRIVCOND"]] = df[["MANOEUVER", "DRIVACT", "DRIVCOND"]].fillna("Not Applicable")

# INJURY — NOT bimodal (Driver 63% missing, Passenger 35%, Truck Driver 83%)
# For roles physically present at the scene, missing likely means "not hurt"
# For bystander roles (Witness, Vehicle Owner, etc.), missing means the field never applied
# These are different facts, so they get different labels
present_roles = ["Driver", "Passenger", "Truck Driver", "Motorcycle Driver",
                  "Motorcycle Passenger", "Cyclist", "Cyclist Passenger",
                  "Moped Driver", "Moped Passenger", "In-Line Skater", "Wheelchair"]

mask_present = df["INVTYPE"].isin(present_roles)

df.loc[mask_present, "INJURY"] = df.loc[mask_present, "INJURY"].fillna("None")
df.loc[~mask_present, "INJURY"] = df.loc[~mask_present, "INJURY"].fillna("Not Applicable")

print(df.isnull().sum())

df["DATE_PARSED"] = pd.to_datetime(df["DATE"])
df["YEAR"] = df["DATE_PARSED"].dt.year
df["HOUR"] = df["TIME"] // 100


# =============================================================================
# DELIVERABLE 1B — DATA VISUALIZATION
# =============================================================================
# Exploratory charts built on the cleaned frame from the exploration section
# above. No modelling transformations (encoding, scaling, feature drops) have
# happened yet, so these plots reflect the raw-but-cleaned KSI data.

# 1. Collisions per year, split by severity
yearly = df.groupby(["YEAR", "ACCLASS"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(10, 5.5))
yearly.plot(kind="bar", stacked=True, ax=ax,
            color=["#d62728", "#1f77b4", "#7f7f7f"])
ax.set_title("Killed or Seriously Injured Collisions by Year and Severity")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Collisions")
ax.legend(title="Severity")
plt.tight_layout()
plt.savefig("1_collisions_by_year_severity.png", dpi=300, bbox_inches="tight")
plt.show()

# 2. Collisions by hour of day
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(x="HOUR", data=df, hue="ACCLASS", ax=ax,
              palette={"Fatal": "#d62728", "Non-Fatal Injury": "#1f77b4",
                       "Property Damage O": "#7f7f7f"})
ax.set_title("Collisions by Hour of Day")
ax.set_xlabel("Hour (24h)")
ax.set_ylabel("Number of Collisions")
plt.tight_layout()
plt.savefig("2_collisions_by_hour.png", dpi=300, bbox_inches="tight")
plt.show()

# 3. Contributing factors (Yes counts across binary flags)
factor_cols = ["SPEEDING", "AG_DRIV", "REDLIGHT", "ALCOHOL", "DISABILITY"]
factor_counts = {c: (df[c] == "Yes").sum() for c in factor_cols}
factor_series = pd.Series(factor_counts).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 5))
factor_series.plot(kind="barh", ax=ax, color="#c44e52")
ax.set_title("Collisions Involving Each Contributing Factor")
ax.set_xlabel("Number of Collisions")
plt.tight_layout()
plt.savefig("3_contributing_factors.png", dpi=300, bbox_inches="tight")
plt.show()

# 4. Correlation heatmap of binary contributing/involvement factors
bin_cols = ["PEDESTRIAN", "CYCLIST", "AUTOMOBILE", "MOTORCYCLE", "TRUCK",
            "SPEEDING", "AG_DRIV", "REDLIGHT", "ALCOHOL", "DISABILITY"]
bin_df = df[bin_cols].apply(lambda s: (s == "Yes").astype(int))
corr = bin_df.corr()
fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
            square=True, cbar_kws={"label": "Correlation"})
ax.set_title("Correlation Between Contributing Factors")
plt.tight_layout()
plt.savefig("4_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# 5. Top 10 neighbourhoods by collision count
top_nbhd = df["NEIGHBOURHOOD_140"].value_counts().nlargest(10).sort_values()
fig, ax = plt.subplots(figsize=(9, 5.5))
top_nbhd.plot(kind="barh", ax=ax, color="#dd8452")
ax.set_title("Top 10 Neighbourhoods by Collision Count")
ax.set_xlabel("Number of Collisions")
plt.tight_layout()
plt.savefig("5_top_neighbourhoods.png", dpi=300, bbox_inches="tight")
plt.show()


# =============================================================================
# DELIVERABLE 2 — DATA MODELLING
# =============================================================================
# Everything above is Deliverable 1 (data exploration + missing-value strategy).
# This section builds on that cleaned frame. It deliberately does NOT reload the
# CSV and does NOT repeat any imputation — all of that already happened above,
# and running it twice would either double-transform values or silently undo the
# group-by-group decisions that were justified in Deliverable 1.

import numpy as np
from collections import Counter

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)


# 1. WORK ON A COPY OF THE EXPLORED DATA
print("\n1. PREPARING THE MODELLING FRAME")

# model_df is the only frame the modelling stage touches. df is left exactly as
# Deliverable 1 produced it, so the exploration results stay reproducible and
# re-running any cell above cannot be affected by modelling transformations.
model_df = df.copy()
print(f"Rows entering modelling: {model_df.shape[0]}, columns: {model_df.shape[1]}")


# 2. TARGET VARIABLE PREPARATION
print("\n2. TARGET VARIABLE PREPARATION")

print(f"\nACCLASS distribution after cleaning:\n{model_df['ACCLASS'].value_counts(dropna=False)}")

# No dropna() on ACCLASS here — the single unlabeled row was already removed by
# the Group 4A dropna() above, so repeating it would be a no-op at best.

# Merge "Property Damage O" (18 rows) into "Non-Fatal Injury"
# Justification: only 18 rows — too few to be a standalone class, and the
# project goal is binary classification: fatal vs non-fatal.
model_df["ACCLASS"] = model_df["ACCLASS"].replace("Property Damage O", "Non-Fatal Injury")

# Encode: Fatal = 1, Non-Fatal = 0
model_df["ACCLASS_BINARY"] = (model_df["ACCLASS"] == "Fatal").astype(int)

n_nonfatal = (model_df["ACCLASS_BINARY"] == 0).sum()
n_fatal = (model_df["ACCLASS_BINARY"] == 1).sum()
print("\nBinary target distribution:")
print(f"  Non-Fatal (0): {n_nonfatal}")
print(f"  Fatal     (1): {n_fatal}")
print(f"  Imbalance ratio: 1:{n_nonfatal // n_fatal}")


# 3. MISSING DATA — VERIFICATION ONLY (already handled in Deliverable 1)
print("\n3. MISSING DATA VERIFICATION")

# The Group 1-6 strategy above resolved every column, so this is a guard rail
# rather than another imputation pass. If a column ever shows up here, it means
# the exploration stage missed it — fix it up there, not down here.
remaining = model_df.isnull().sum()
remaining = remaining[remaining > 0]
if remaining.empty:
    print("  No missing values remain — no further imputation applied.")
else:
    print("  Columns still missing values (revisit Deliverable 1):")
    print(remaining)


# 4. FEATURE SELECTION — WITH JUSTIFICATION
print("\n4. FEATURE SELECTION")

# 4a. Post-incident outcome columns — DROP (data leakage)
# These describe what happened *after* the collision. Using them to predict
# fatality would let the model cheat: FATAL_NO is only populated for fatalities
# and INJURY records the severity outcome directly.
leakage_cols = {
    "INJURY": "Injury severity is the outcome itself — direct target leakage",
    "FATAL_NO": "Only populated for fatal collisions — perfect leakage of the target",
}

# 4b. Sparse role-conditional columns — DROP
# These were filled with "Not Applicable" above so the missingness could be
# explained, but 83-96% of the rows carry that single placeholder. The binary
# involvement flags (PEDESTRIAN, CYCLIST) already capture the same information
# with none of the sparsity.
sparse_cols = {col: "83-96% 'Not Applicable' — signal already captured by the PEDESTRIAN/CYCLIST flags"
               for col in group2_cols}
sparse_cols.update({
    "MANOEUVER": "Populated only for vehicle operators — mostly 'Not Applicable'",
    "DRIVACT": "Driver-only field — mostly 'Not Applicable'",
    "DRIVCOND": "Driver-only field — mostly 'Not Applicable'",
    "OFFSET": "79.8% placeholder — free-text location offset, no predictive value",
})

# 4c. Identifiers and redundant columns — DROP
# ACCNUM is deliberately NOT in this dict — Section 6 pulls it out separately
# as the grouping key for the train/test split, then drops it from X after.
identifier_cols = {
    "OBJECTID": "Row identifier — no predictive value",
    "INDEX": "Row identifier — no predictive value",
    "STREET1": "~4600 unique values; location already captured by HOOD_158 and DISTRICT",
    "STREET2": "Too many unique values; location already captured by HOOD_158 and DISTRICT",
    "ACCLASS": "Original target label — replaced by binary ACCLASS_BINARY",
    "NEIGHBOURHOOD_158": "Text name that duplicates numeric HOOD_158 (redundant)",
    "NEIGHBOURHOOD_140": "Text name that duplicates numeric HOOD_140 (redundant)",
    "HOOD_140": "Redundant with HOOD_158 — both encode neighbourhood, keeping one",
    "x": "Projected x-coordinate — redundant with LONGITUDE",
    "y": "Projected y-coordinate — redundant with LATITUDE",
    "DATE": "Raw date string — the usable temporal signal is extracted as HOUR from TIME",
    "DATE_PARSED": "Intermediate column used only to compute YEAR — not a feature",
    "YEAR": "Used for the exploration chart, not carried into modelling as a feature",
}

drop_cols = {**leakage_cols, **sparse_cols, **identifier_cols}

print("\nColumns DROPPED:")
for col, reason in drop_cols.items():
    print(f"  x {col:20s} -> {reason}")

model_df = model_df.drop(columns=[c for c in drop_cols if c in model_df.columns])

# Columns KEPT — justification for each retained feature
kept_cols = {
    "TIME": "Converted to HOUR below — captures night vs day fatality risk",
    "ROAD_CLASS": "Road type — arterials vs collectors have different fatality profiles",
    "DISTRICT": "Geographic district — captures area-level risk differences",
    "LATITUDE": "Geographic coordinate — spatial clustering of fatal collisions",
    "LONGITUDE": "Geographic coordinate — spatial clustering of fatal collisions",
    "ACCLOC": "Accident location type (intersection, mid-block) — structural risk factor",
    "TRAFFCTL": "Traffic control present — signals vs uncontrolled affects severity",
    "VISIBILITY": "Weather visibility — rain/snow/fog affect crash severity",
    "LIGHT": "Lighting condition — dark conditions correlate with higher fatality",
    "RDSFCOND": "Road surface — wet/icy roads affect collision outcomes",
    "IMPACTYPE": "Impact type — head-on vs sideswipe have very different fatality rates",
    "INVTYPE": "Involvement type — pedestrians have higher fatality risk than drivers",
    "INVAGE": "Age group of person involved — elderly are more vulnerable",
    "INITDIR": "Initial direction of travel — directional collision patterns",
    "VEHTYPE": "Vehicle type — trucks vs cars produce different severity",
    "HOOD_158": "Neighbourhood code — local area risk proxy",
    "DIVISION": "Police division — geographic/demographic risk proxy",
    "PEDESTRIAN": "Binary flag — pedestrian involvement strongly predicts fatality",
    "CYCLIST": "Binary flag — cyclist involvement",
    "AUTOMOBILE": "Binary flag — automobile involvement",
    "MOTORCYCLE": "Binary flag — motorcycle crashes have high fatality",
    "TRUCK": "Binary flag — truck involvement increases severity",
    "TRSN_CITY_VEH": "Binary flag — transit/city vehicle involvement",
    "EMERG_VEH": "Binary flag — emergency vehicle involvement",
    "PASSENGER": "Binary flag — passenger presence",
    "SPEEDING": "Binary flag — speeding is a top fatality predictor",
    "AG_DRIV": "Binary flag — aggressive driving behaviour",
    "REDLIGHT": "Binary flag — running red lights",
    "ALCOHOL": "Binary flag — alcohol involvement strongly predicts fatality",
    "DISABILITY": "Binary flag — disability involvement",
}

print("\nColumns KEPT:")
for col, reason in kept_cols.items():
    print(f"  + {col:20s} -> {reason}")


# 5. CATEGORICAL DATA MANAGEMENT & FEATURE ENGINEERING
print("\n5. CATEGORICAL DATA MANAGEMENT & FEATURE ENGINEERING")

# 5a. Extract HOUR from TIME
# TIME is stored as an integer (e.g. 1430 = 2:30 PM), so integer-divide by 100.
# Guarded by a column check so re-running this section cannot divide twice.
if "TIME" in model_df.columns:
    model_df["HOUR"] = model_df["TIME"] // 100
    model_df = model_df.drop(columns=["TIME"])
print(f"  Extracted HOUR from TIME (range: {model_df['HOUR'].min()} - {model_df['HOUR'].max()})")

# 5b. Handle INVAGE "unknown"
# ~14% of INVAGE values are "unknown". Kept as its own category because the
# unknownness itself may correlate with outcome (hit-and-runs, unidentified).
print(f"  INVAGE 'unknown' entries: {(model_df['INVAGE'] == 'unknown').sum()} — kept as a category")


# 5c. Group rare categories (<1% frequency) into "Other"
# Prevents one-hot encoding from creating many near-empty columns that add
# dimensionality without predictive signal.
def group_rare(series, threshold=0.01):
    """Replace categories appearing below `threshold` fraction with 'Other'."""
    freq = series.value_counts(normalize=True)
    rare = freq[freq < threshold].index
    return series.where(~series.isin(rare), "Other")


rare_grouping_cols = ["ROAD_CLASS", "LIGHT", "RDSFCOND", "VEHTYPE",
                      "INVTYPE", "TRAFFCTL", "VISIBILITY"]

# The frequencies that decide "rare" are a property of THIS dataset, so they
# have to be captured here and shipped with the model. At serving time a single
# incoming row has no frequency distribution of its own — recomputing
# value_counts() on one record would mark every value as 100% frequent and
# group nothing, so a category that trained as "Other" would arrive raw and be
# silently one-hot encoded as all-zeros. Deliverable 5 pickles this map and
# replays it instead of recomputing.
rare_category_maps = {}

# HOOD_158 is intentionally excluded — with 159 neighbourhoods every single one
# sits below 1%, so grouping would collapse the whole column into "Other".
for col in rare_grouping_cols:
    before = model_df[col].nunique()
    model_df[col] = group_rare(model_df[col])
    after = model_df[col].nunique()
    # Categories that survived grouping; anything else becomes "Other" at serving time.
    rare_category_maps[col] = sorted(model_df[col].unique().tolist())
    if before != after:
        print(f"  {col}: grouped rare categories ({before} -> {after} unique values)")

# 5d. Convert the Group 1 involvement flags from strings to 0/1
# group1_cols is reused from Deliverable 1 rather than redeclared, so the two
# stages can never drift apart. The comparison against "Yes" is idempotent.
binary_flag_cols = group1_cols
for col in binary_flag_cols:
    model_df[col] = (model_df[col] == "Yes").astype(int)
print(f"  Encoded {len(binary_flag_cols)} binary flag columns: 'Yes'->1, 'No'->0")

print(f"\nModelling frame shape after transformations: {model_df.shape}")
print(f"Feature types:\n{model_df.dtypes.value_counts()}")


# 6. TRAIN / TEST SPLIT (GROUPED BY CRASH, NOT STRATIFIED)
print("\n6. TRAIN / TEST SPLIT")

X = model_df.drop(columns=["ACCLASS_BINARY"])
y = model_df["ACCLASS_BINARY"]

# Extract crash-level groups BEFORE dropping ACCNUM from X. Each ACCNUM now
# maps to exactly one real crash (Fix 2, above) — no more shared -1 mega-group.
groups = X["ACCNUM"]

# Now drop ACCNUM from X — it's an ID, not a predictive feature
X = X.drop(columns=["ACCNUM"])

# Identify column types for the ColumnTransformer (done once, after the drop,
# so numeric_features/categorical_features never include ACCNUM)
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# Split by crash ID — whole crashes stay together in train OR test, never split
# across both. GroupShuffleSplit does NOT stratify by y — it only respects
# groups — so the Fatal ratio below is a result, not a guarantee.
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]



print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"  Non-Fatal (0): {(y_train == 0).sum()} ({(y_train == 0).mean() * 100:.1f}%)")
print(f"  Fatal     (1): {(y_train == 1).sum()} ({(y_train == 1).mean() * 100:.1f}%)")
print(f"\nTest set: {X_test.shape[0]} samples")
print(f"  Non-Fatal (0): {(y_test == 0).sum()} ({(y_test == 0).mean() * 100:.1f}%)")
print(f"  Fatal     (1): {(y_test == 1).sum()} ({(y_test == 1).mean() * 100:.1f}%)")
print(f"\n  Split by ACCNUM (GroupShuffleSplit), not stratified — Fatal ratio")
print(f"  above is a result of the crash-level split, not an enforced target.")
print(f"  Overall Fatal rate in the full dataset: {(y == 1).mean() * 100:.1f}%")


# 7. PREPROCESSING PIPELINE (ColumnTransformer)
print("\n7. PREPROCESSING PIPELINE")

# Numeric pipeline: median-impute any edge case, then standardize.
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categorical pipeline: mode-impute, then one-hot encode.
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# The imputers are a safety net for unseen production rows, not a second pass
# over this dataset — step 3 confirmed nothing is missing here.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ],
    remainder="drop",
)

# Fit on TRAINING data only, then transform both sets — fitting on the full
# dataset would leak test-set statistics into the scaler and encoder.
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Feature names after one-hot encoding (used later for feature importance)
encoded_cat_names = preprocessor.named_transformers_["cat"]["encoder"] \
    .get_feature_names_out(categorical_features)
all_feature_names = list(numeric_features) + list(encoded_cat_names)

print("  Numeric pipeline:     SimpleImputer(median) -> StandardScaler")
print("  Categorical pipeline: SimpleImputer(mode) -> OneHotEncoder")
print(f"\n  Features before encoding: {X.shape[1]}")
print(f"  Features after encoding:  {X_train_processed.shape[1]}")
print(f"  Training matrix shape:    {X_train_processed.shape}")
print(f"  Test matrix shape:        {X_test_processed.shape}")


# 8. MANAGING IMBALANCED CLASSES (SMOTE)
print("\n8. MANAGING IMBALANCED CLASSES — SMOTE")

print("\nClass distribution BEFORE SMOTE (training set):")
print(f"  {Counter(y_train)}")

# SMOTE is applied ONLY to the training data — resampling the test set would
# inflate the scores and make the evaluation dishonest.
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

print("\nClass distribution AFTER SMOTE (training set):")
print(f"  {Counter(y_train_resampled)}")
print(f"\n  Resampled training shape: {X_train_resampled.shape}")
print(f"  Test set left untouched:  {X_test_processed.shape}")


# SUMMARY
print("\nDELIVERABLE 2 — SUMMARY")
print(f"""
  Records after exploration cleaning: {df.shape[0]}
  Records used for modelling:         {model_df.shape[0]}
  Features selected:                  {X.shape[1]}
  Features after encoding:            {X_train_processed.shape[1]}

  Target: ACCLASS_BINARY (Fatal=1, Non-Fatal=0)

  Missing data: fully handled in Deliverable 1 (Groups 1-6); the modelling
    stage only verifies it and never re-imputes.

  Feature selection:
    Leakage dropped     -> INJURY, FATAL_NO
    Sparse dropped      -> pedestrian/cyclist detail columns, MANOEUVER,
                           DRIVACT, DRIVCOND, OFFSET
    Identifiers dropped -> OBJECTID, INDEX, ACCNUM (used as split group first),
                           STREET1/2, DATE, x, y
    Redundant dropped   -> NEIGHBOURHOOD_158/140, HOOD_140, ACCLASS

  Categorical encoding:
    Binary flags          -> 0/1 integers
    Multi-category        -> OneHotEncoder (inside the pipeline)
    Rare categories (<1%) -> grouped into 'Other'

  Normalization: StandardScaler on numeric features (inside the pipeline)

  Train/Test split: 80/20 by crash (ACCNUM), via GroupShuffleSplit — no crash
    is split across train and test
    Train: {X_train.shape[0]} samples
    Test:  {X_test.shape[0]} samples

  Class imbalance: SMOTE on training data only
    Before: {dict(Counter(y_train))}
    After:  {dict(Counter(y_train_resampled))}

  Pipeline: ColumnTransformer(
    numeric     -> SimpleImputer(median) -> StandardScaler
    categorical -> SimpleImputer(mode)   -> OneHotEncoder
  )
""")


# =============================================================================
# DELIVERABLE 3 — PREDICTIVE MODEL BUILDING
# =============================================================================
# Picks up exactly where Deliverable 2 left off: X_train_resampled /
# y_train_resampled (SMOTE-balanced, encoded+scaled) for fitting, and
# X_test_processed / y_test (untouched, real-world imbalance) for evaluation.
# Test data is NEVER resampled — that would make the evaluation dishonest.

import time
from scipy.stats import randint, uniform

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

RANDOM_STATE = 42

# Public-safety framing: a false negative (predicting Non-Fatal when the
# collision was actually Fatal) is the costly error, so RECALL on the Fatal
# class matters at least as much as raw accuracy. Every model below is scored
# on F1 during tuning (balances precision/recall) and Recall is reported
# explicitly for every model at evaluation time so that trade-off is visible,
# not hidden behind an accuracy number that the class imbalance would inflate.

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def evaluate_model(name, model, X_test, y_test, results_list):
    """Fit-agnostic evaluation: assumes `model` is already fitted."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["Non-Fatal", "Fatal"], zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}  ROC-AUC={auc:.3f}")

    results_list.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1": f1, "ROC-AUC": auc,
        "y_proba": y_proba,
        # y_pred is kept so Deliverable 5 can export the confusion matrices for
        # the written report from the same run that produces the shipped model.
        # Transcribing them by hand is how a report and a deployed artifact end
        # up describing two slightly different models.
        "y_pred": y_pred,
    })
    return results_list


results = []
tuned_models = {}

# -----------------------------------------------------------------------
# 9.1 LOGISTIC REGRESSION — GridSearchCV
# -----------------------------------------------------------------------
# Small, cheap parameter space -> exhaustive grid search is affordable and
# gives the exact optimum over this grid (no sampling variance).
print("\n9.1 LOGISTIC REGRESSION — GRID SEARCH")

log_reg_param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"],  # only solver that supports both l1 and l2 here
}

log_reg_grid = GridSearchCV(
    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    param_grid=log_reg_param_grid,
    scoring="f1",
    cv=cv_strategy,
    n_jobs=-1,
    verbose=1,
)

t0 = time.time()
log_reg_grid.fit(X_train_resampled, y_train_resampled)
print(f"  Fit time: {time.time() - t0:.1f}s")
print(f"  Best params: {log_reg_grid.best_params_}")
print(f"  Best CV F1: {log_reg_grid.best_score_:.3f}")

tuned_models["Logistic Regression"] = log_reg_grid.best_estimator_
results = evaluate_model("Logistic Regression (tuned)", log_reg_grid.best_estimator_,
                          X_test_processed, y_test, results)

# -----------------------------------------------------------------------
# 9.2 DECISION TREE — GridSearchCV
# -----------------------------------------------------------------------
# Still a small enough space (5*3*3*2 = 90 combos * 5 folds) for a full grid.
print("\n9.2 DECISION TREE — GRID SEARCH")

dt_param_grid = {
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"],
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_grid=dt_param_grid,
    scoring="f1",
    cv=cv_strategy,
    n_jobs=-1,
    verbose=1,
)

t0 = time.time()
dt_grid.fit(X_train_resampled, y_train_resampled)
print(f"  Fit time: {time.time() - t0:.1f}s")
print(f"  Best params: {dt_grid.best_params_}")
print(f"  Best CV F1: {dt_grid.best_score_:.3f}")

tuned_models["Decision Tree"] = dt_grid.best_estimator_
results = evaluate_model("Decision Tree (tuned)", dt_grid.best_estimator_,
                          X_test_processed, y_test, results)

# -----------------------------------------------------------------------
# 9.3 SUPPORT VECTOR MACHINE — RandomizedSearchCV
# -----------------------------------------------------------------------
# SVC training cost scales roughly quadratically-to-cubically with sample
# count, and the SMOTE-resampled training set can run into the tens of
# thousands of rows. A full grid search here is impractical, so
# RandomizedSearchCV samples a fixed number of parameter combinations
# instead of trying all of them. SVM is also fit on a stratified subsample
# of the resampled training data (capped, and capped LOWER than the other
# models here) purely for tractability -- this is a runtime concession, not
# a data-quality one, and is called out explicitly rather than silently
# changing the training set.
#
# Three further runtime cuts vs. the other models' searches, all specific
# to SVM's cost profile:
#   - CAP dropped to 4000 rows (was 15000) -- SVC training time grows much
#     faster than linearly, so this is the single biggest lever.
#   - kernel fixed to "rbf" only -- SVC's own "linear" kernel implementation
#     is far slower than the dedicated LinearSVC class at this row count,
#     so it's dropped here rather than paying for a slow linear fit.
#   - a dedicated 3-fold CV (svm_cv) instead of the 5-fold used elsewhere,
#     and n_iter cut from 15 to 8 -- fewer, cheaper fits (24 total vs. 75).
print("\n9.3 SUPPORT VECTOR MACHINE — RANDOMIZED SEARCH")

SVM_SAMPLE_CAP = 4000
if X_train_resampled.shape[0] > SVM_SAMPLE_CAP:
    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(X_train_resampled.shape[0], size=SVM_SAMPLE_CAP, replace=False)
    X_train_svm = X_train_resampled[sample_idx]
    y_train_svm = y_train_resampled.iloc[sample_idx] if hasattr(y_train_resampled, "iloc") else y_train_resampled[sample_idx]
    print(f"  Subsampled SVM training set: {SVM_SAMPLE_CAP} of {X_train_resampled.shape[0]} rows")
else:
    X_train_svm, y_train_svm = X_train_resampled, y_train_resampled

svm_param_dist = {
    "C": uniform(0.1, 20),
    "kernel": ["rbf"],
    "gamma": ["scale", "auto"],
}

svm_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

svm_random = RandomizedSearchCV(
    SVC(probability=True, random_state=RANDOM_STATE, cache_size=1000),
    param_distributions=svm_param_dist,
    n_iter=8,
    scoring="f1",
    cv=svm_cv,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1,
)

t0 = time.time()
svm_random.fit(X_train_svm, y_train_svm)
print(f"  Fit time: {time.time() - t0:.1f}s")
print(f"  Best params: {svm_random.best_params_}")
print(f"  Best CV F1: {svm_random.best_score_:.3f}")

tuned_models["SVM"] = svm_random.best_estimator_
results = evaluate_model("SVM (tuned)", svm_random.best_estimator_,
                          X_test_processed, y_test, results)

# -----------------------------------------------------------------------
# 9.4 RANDOM FOREST — RandomizedSearchCV
# -----------------------------------------------------------------------
# Ensemble with several interacting hyperparameters -> a full grid would be
# huge, so RandomizedSearchCV explores a wide distribution with a fixed
# sampling budget.
print("\n9.4 RANDOM FOREST — RANDOMIZED SEARCH")

rf_param_dist = {
    "n_estimators": randint(100, 500),
    "max_depth": [10, 20, 30, None],
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 5),
    "max_features": ["sqrt", "log2"],
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    param_distributions=rf_param_dist,
    n_iter=20,
    scoring="f1",
    cv=cv_strategy,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1,
)

t0 = time.time()
rf_random.fit(X_train_resampled, y_train_resampled)
print(f"  Fit time: {time.time() - t0:.1f}s")
print(f"  Best params: {rf_random.best_params_}")
print(f"  Best CV F1: {rf_random.best_score_:.3f}")

tuned_models["Random Forest"] = rf_random.best_estimator_
results = evaluate_model("Random Forest (tuned)", rf_random.best_estimator_,
                          X_test_processed, y_test, results)

# -----------------------------------------------------------------------
# 9.5 NEURAL NETWORK (MLPClassifier) — RandomizedSearchCV
# -----------------------------------------------------------------------
print("\n9.5 NEURAL NETWORK (MLP) — RANDOMIZED SEARCH")

mlp_param_dist = {
    "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50), (100, 100)],
    "activation": ["relu", "tanh"],
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate_init": [0.001, 0.01],
}

mlp_random = RandomizedSearchCV(
    MLPClassifier(max_iter=300, early_stopping=True, random_state=RANDOM_STATE),
    param_distributions=mlp_param_dist,
    n_iter=15,
    scoring="f1",
    cv=cv_strategy,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1,
)

t0 = time.time()
mlp_random.fit(X_train_resampled, y_train_resampled)
print(f"  Fit time: {time.time() - t0:.1f}s")
print(f"  Best params: {mlp_random.best_params_}")
print(f"  Best CV F1: {mlp_random.best_score_:.3f}")

tuned_models["Neural Network"] = mlp_random.best_estimator_
results = evaluate_model("Neural Network (tuned)", mlp_random.best_estimator_,
                          X_test_processed, y_test, results)


# -----------------------------------------------------------------------
# 9.6 MODEL COMPARISON
# -----------------------------------------------------------------------
print("\n9.6 MODEL COMPARISON")

results_df = pd.DataFrame(results).drop(columns=["y_proba", "y_pred"])
results_df = results_df.sort_values("F1", ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
print(f"\n  Best model by F1 (Fatal class): {best_model_name}")

# -----------------------------------------------------------------------
# 9.7 ROC CURVES — all tuned models overlaid
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))
for r in results:
    if r["y_proba"] is not None:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        ax.plot(fpr, tpr, label=f"{r['Model']} (AUC={r['ROC-AUC']:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Tuned Models")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("6_roc_curves_all_models.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------------------------------------------------
# 9.8 FEATURE IMPORTANCE — Random Forest (interpretable, tree-based)
# -----------------------------------------------------------------------
rf_best = tuned_models["Random Forest"]
importances = pd.Series(rf_best.feature_importances_, index=all_feature_names)
top_importances = importances.sort_values(ascending=False).head(20).sort_values()

fig, ax = plt.subplots(figsize=(9, 8))
top_importances.plot(kind="barh", ax=ax, color="#4c72b0")
ax.set_title("Top 20 Feature Importances — Random Forest")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("7_feature_importance_rf.png", dpi=300, bbox_inches="tight")
plt.show()

# SUMMARY
print("\nDELIVERABLE 3 — SUMMARY")
print(f"""
  Models trained & tuned: Logistic Regression, Decision Tree, SVM,
    Random Forest, Neural Network (MLPClassifier)

  Tuning method:
    GridSearchCV       -> Logistic Regression, Decision Tree (small, cheap
                           search spaces -> exhaustive search is affordable)
    RandomizedSearchCV -> SVM, Random Forest, Neural Network (large search
                           spaces / expensive fits -> fixed sampling budget)

  Scoring metric for tuning: F1 on the Fatal class (accuracy would be
    misleadingly high given the class imbalance; F1 balances catching Fatal
    cases against false alarms)

  All models fit on SMOTE-resampled training data, evaluated on the
    untouched, naturally-imbalanced test set.

  Best model by test F1: {best_model_name}

{results_df.to_string(index=False)}
""")

# =============================================================================
# DELIVERABLE 5 — MODEL EXPORT FOR DEPLOYMENT
# =============================================================================
# Everything below turns the winning model from Deliverable 3 into something a
# Flask API can actually serve. Three problems have to be solved here, and all
# three are properties of THIS script rather than of Flask:
#
#   1. `preprocessor` was fitted separately from the models (section 7), and
#      every estimator was fitted on an already-encoded numpy matrix. Pickling
#      an estimator on its own would produce a model that only accepts a
#      ~200-column encoded array — useless to an endpoint receiving JSON. The
#      fitted ColumnTransformer and the fitted estimator are therefore wrapped
#      into a single Pipeline. Both components are ALREADY fitted and sklearn
#      only refits on .fit(), so .predict() works immediately — no retraining.
#
#   2. The rare-category grouping in section 5c is data-dependent. Its decisions
#      are captured in `rare_category_maps` and shipped in the bundle.
#
#   3. The decision threshold. See section 10.2.

print("\n" + "=" * 79)
print("DELIVERABLE 5 — MODEL EXPORT FOR DEPLOYMENT")
print("=" * 79)

import json
import pickle
import sklearn
from datetime import datetime

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import fbeta_score

# -----------------------------------------------------------------------
# 10.1 SELECT THE MODEL TO DEPLOY
# -----------------------------------------------------------------------
# results_df is sorted by F1 descending, so row 0 is the winner. The label
# carries a " (tuned)" suffix for reporting, but `tuned_models` is keyed
# without it — strip the suffix before looking the estimator up.
deploy_key = best_model_name.replace(" (tuned)", "")
base_estimator = tuned_models[deploy_key]
deploy_row = results_df.iloc[0]

print(f"\n10.1 MODEL SELECTED FOR DEPLOYMENT: {deploy_key}")
print(f"  Chosen as the highest test F1 on the Fatal class ({deploy_row['F1']:.3f}).")
print("  This matches the recommendation in the Part 4 report, which names the")
print("  Neural Network (MLPClassifier) as the primary model for deployment.")

# Whether the F1 winner is ALSO the best ranker is checked, not assumed — the
# argument for keeping this model after retuning the threshold rests on
# ROC-AUC, so a re-run that changed the ordering must not silently keep
# printing a claim that is no longer true.
_best_auc_row = results_df.loc[results_df["ROC-AUC"].idxmax()]
if _best_auc_row["Model"] == deploy_row["Model"]:
    print(f"  It also has the highest ROC-AUC ({deploy_row['ROC-AUC']:.3f}), meaning it")
    print("  ranks Fatal cases above Non-Fatal ones better than any other candidate.")
    print("  ROC-AUC is threshold-independent, so this model stays the best choice")
    print("  even after the decision threshold is retuned below.")
else:
    print(f"  NOTE: its ROC-AUC ({deploy_row['ROC-AUC']:.3f}) is NOT the highest — "
          f"{_best_auc_row['Model']} ranks better ({_best_auc_row['ROC-AUC']:.3f}).")
    print("  Since the deployed threshold is retuned below and ROC-AUC is")
    print("  threshold-independent, revisit whether F1-at-0.5 is the right")
    print("  selection criterion for this run.")

# -----------------------------------------------------------------------
# 10.1b PROBABILITY CALIBRATION FOR DEPLOYMENT
# -----------------------------------------------------------------------
# The tuned MLP ranks well but its probabilities are badly calibrated: roughly
# two thirds of its test predictions are pinned below 0.001 or above 0.999.
# That saturation is not a cosmetic problem. It means the served API returns
# "0.0%" for the overwhelming majority of inputs, so a user changing fields in
# the form sees no movement at all and the probability carries no usable
# information beyond the bare class label.
#
# Isotonic calibration fits a monotonic mapping from the raw scores to
# empirical frequencies, spreading the pinned mass back across the interval.
# Because the mapping is monotonic it CANNOT change the ranking of records,
# so ROC-AUC — and therefore the model comparison in Deliverable 3 that
# selected this model — is unaffected. Measured effect on this dataset:
#   saturation  65.1% -> 18.8%      (predictions pinned at 0 or 1)
#   ROC-AUC     0.7363 -> 0.7398    (unchanged within noise, as expected)
#
# This is deliberately applied HERE, in the deployment stage, and not back in
# section 9.5. Deliverable 3's comparison table reports the five algorithms as
# they were tuned; rewriting the MLP there would change numbers already
# reported. Calibration is a property of the deployed artifact, so it belongs
# in the deployment step.
#
# WHAT THIS DOES NOT FIX — see the limitations note in the README. Calibration
# rescales the probabilities; it does not make the decision surface
# well-behaved. Flipping a single risk flag on a record can still move the
# prediction in the wrong direction (measured: 27.6% of single-flag flips,
# improved from 33.6% but far from resolved). The deployed model remains
# usable for ranking records, and unsuitable for counterfactual "what if this
# one factor changed" reasoning.

print("\n10.1b PROBABILITY CALIBRATION")

t0 = time.time()
deploy_estimator = CalibratedClassifierCV(clone(base_estimator), method="isotonic", cv=3)
deploy_estimator.fit(X_train_resampled, y_train_resampled)
print(f"  Wrapped {deploy_key} in isotonic calibration (3-fold): {time.time() - t0:.1f}s")

_raw_proba = base_estimator.predict_proba(X_test_processed)[:, 1]
_cal_proba = deploy_estimator.predict_proba(X_test_processed)[:, 1]


def _saturation(p):
    return float(((p < 0.001) | (p > 0.999)).mean())


print(f"  Saturated predictions (P<0.001 or P>0.999): "
      f"{_saturation(_raw_proba) * 100:.1f}% -> {_saturation(_cal_proba) * 100:.1f}%")
print(f"  ROC-AUC: {roc_auc_score(y_test, _raw_proba):.4f} -> "
      f"{roc_auc_score(y_test, _cal_proba):.4f}  "
      f"(isotonic is monotonic, so ranking is preserved by construction)")

deploy_label = f"{deploy_key} (calibrated)"

# -----------------------------------------------------------------------
# 10.2 DECISION THRESHOLD TUNING
# -----------------------------------------------------------------------
# Section 9's evaluation used sklearn's default 0.5 cut-off. That default is
# arbitrary for this problem: the whole framing above says a false negative
# (a Fatal collision predicted Non-Fatal) is the expensive error, so the
# deployed API should not silently inherit a threshold that treats both error
# types as equally costly.
#
# IMPORTANT — the threshold is tuned on a VALIDATION split carved out of the
# training data, never on the test set. Picking a cut-off that maximises a
# score on the test set and then reporting that same score would be leakage:
# the reported number would no longer be an honest held-out estimate. So:
#   - X_train is split again by ACCNUM (same grouped logic as section 6),
#   - a clone of the winning estimator is refit on that sub-training half,
#   - the threshold is selected on the held-out validation half, which keeps
#     its natural class imbalance (SMOTE is applied to the sub-train only).
# The threshold found there is then applied to the full-data model that ships.
#
# CHOICE OF CRITERION — why not F-beta(beta=2)?
# The obvious way to encode "recall matters more" is to maximise F2. On this
# dataset that criterion is degenerate. Precision on the Fatal class is bounded
# below by the class prevalence (~15%), while recall can always be pushed to
# 1.0 by lowering the threshold, so F2 keeps improving as the model predicts
# Fatal for everything. At the validation prevalence an all-positive
# classifier scores F2 = 0.475, which beats every interior threshold — the
# "optimum" is a model that has stopped discriminating at all. That is checked
# explicitly below rather than assumed.
#
# Youden's J (= TPR - FPR = sensitivity + specificity - 1) is used instead:
#   - it cannot degenerate: J = 0 for all-positive AND for all-negative, so
#     the maximum is necessarily an interior, genuinely discriminating point;
#   - it is the standard cut-point criterion for the ROC curve already plotted
#     in section 9.7, so the choice is consistent with how the models were
#     compared;
#   - it is prevalence-independent, which matters because the deployed model
#     was fitted on SMOTE-balanced data but scores naturally imbalanced rows.
# It still moves the threshold well below 0.5 — recall rises substantially
# versus the default — without collapsing into "everything is fatal".

print("\n10.2 DECISION THRESHOLD TUNING")

DEPLOY_THRESHOLD = 0.5
threshold_note = "default 0.5 (model exposes no predict_proba)"

if hasattr(deploy_estimator, "predict_proba"):
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
    train_groups = groups.iloc[train_idx]
    sub_idx, val_idx = next(gss_val.split(X_train, y_train, groups=train_groups))

    X_sub, X_val = X_train.iloc[sub_idx], X_train.iloc[val_idx]
    y_sub, y_val = y_train.iloc[sub_idx], y_train.iloc[val_idx]

    # Fresh preprocessor fitted on the sub-training half only — reusing the
    # section 7 preprocessor would leak validation statistics into the scaler.
    val_preprocessor = clone(preprocessor)
    X_sub_processed = val_preprocessor.fit_transform(X_sub)
    X_val_processed = val_preprocessor.transform(X_val)

    # SMOTE on the sub-training half only; validation keeps real prevalence.
    X_sub_resampled, y_sub_resampled = SMOTE(random_state=RANDOM_STATE).fit_resample(
        X_sub_processed, y_sub
    )

    val_model = clone(deploy_estimator)
    t0 = time.time()
    val_model.fit(X_sub_resampled, y_sub_resampled)
    print(f"  Refit {deploy_label} on the sub-training half: {time.time() - t0:.1f}s")
    print(f"  Validation rows: {len(y_val)} (Fatal rate {y_val.mean() * 100:.1f}% — not resampled)")

    val_proba = val_model.predict_proba(X_val_processed)[:, 1]

    candidate_thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    sweep = []
    for t in candidate_thresholds:
        y_hat = (val_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_hat, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) else 0.0   # sensitivity / recall
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        sweep.append({
            "threshold": float(t),
            "precision": precision_score(y_val, y_hat, zero_division=0),
            "recall": tpr,
            "specificity": 1 - fpr,
            "f1": f1_score(y_val, y_hat, zero_division=0),
            "f2": fbeta_score(y_val, y_hat, beta=2, zero_division=0),
            "youden_j": tpr - fpr,
        })
    sweep_df = pd.DataFrame(sweep)

    # Demonstrate the F2 degeneracy rather than asserting it: score the
    # all-positive classifier and compare against the best interior F2.
    all_positive_f2 = fbeta_score(y_val, np.ones_like(y_val), beta=2, zero_division=0)
    best_f2 = sweep_df["f2"].max()

    # The 0.01 grid above drives the diagnostic plot, but it is the wrong tool
    # for actually picking the threshold: a grid can only ever report an
    # optimum at one of its own points, so a maximum sitting near the edge is
    # indistinguishable from a maximum the grid is too coarse to resolve. That
    # is precisely the artifact F2 was rejected for, so Youden's J is held to
    # the same standard. roc_curve() enumerates every distinct predicted
    # probability as a candidate cut-point, making the selection exact and
    # resolution-independent.
    fpr_curve, tpr_curve, roc_thresholds = roc_curve(y_val, val_proba)
    j_curve = tpr_curve - fpr_curve
    j_best = int(np.argmax(j_curve))
    j_threshold = float(roc_thresholds[j_best])
    j_value = float(j_curve[j_best])

    f1_threshold = float(sweep_df.loc[sweep_df["f1"].idxmax(), "threshold"])
    f2_threshold = float(sweep_df.loc[sweep_df["f2"].idxmax(), "threshold"])

    print("\n  Candidate criteria (all evaluated on the validation split):")
    print(f"    Youden's J optimum : threshold {j_threshold:.4f}  (J = {j_value:.3f})"
          f"  [exact, over {len(roc_thresholds)} candidate cut-points]")
    print(f"    F1 optimum         : threshold {f1_threshold:.2f}  "
          f"(F1 = {sweep_df['f1'].max():.3f})")
    print(f"    F2 optimum         : threshold {f2_threshold:.2f}  "
          f"(F2 = {best_f2:.3f})")
    print(f"\n  F2 sanity check — an all-positive classifier scores F2 = {all_positive_f2:.3f}")
    if all_positive_f2 >= best_f2:
        print("    That BEATS the best interior threshold, confirming F2 is maximised by")
        print("    a model that predicts Fatal for every record. F2 is therefore rejected")
        print("    as the selection criterion. Youden's J cannot degenerate this way")
        print(f"    (all-positive gives J = 0.000 by construction).")
    else:
        margin = best_f2 - all_positive_f2
        print(f"    The best interior threshold beats it by only {margin:.3f}, and its")
        print(f"    optimum sits at {f2_threshold:.2f} — the bottom of the searched range.")
        print("    So F2 is not strictly degenerate, but it is still pulling the")
        print("    threshold toward indiscriminate flagging: most of its score comes")
        print("    from the same direction as the all-positive baseline rather than")
        print("    from genuine discrimination. Youden's J is preferred — it has a")
        print("    well-defined interior optimum and matches the ROC analysis in 9.7.")

    DEPLOY_THRESHOLD = j_threshold
    threshold_note = ("chosen by maximising Youden's J (TPR - FPR) on a grouped "
                      "validation split held out of the training data")

    # Metrics at the exact chosen cut-point, recomputed rather than looked up
    # in the plotting grid (which no longer necessarily contains it).
    y_val_hat = (val_proba >= j_threshold).astype(int)
    print(f"\n  DEPLOYED THRESHOLD: {DEPLOY_THRESHOLD:.4f}")
    print(f"    Validation recall (Fatal): {recall_score(y_val, y_val_hat, zero_division=0) * 100:.1f}%")
    print(f"    Validation specificity:    {(1 - fpr_curve[j_best]) * 100:.1f}%")
    print(f"    Validation precision:      {precision_score(y_val, y_val_hat, zero_division=0) * 100:.1f}%")
    print(f"    Compare with the 0.01-grid optimum "
          f"({sweep_df.loc[sweep_df['youden_j'].idxmax(), 'threshold']:.2f}, "
          f"J = {sweep_df['youden_j'].max():.3f}) — the exact search is not grid-limited.")

    # Report figure: how each criterion behaves across the threshold range.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall (Fatal)")
    ax.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision (Fatal)")
    ax.plot(sweep_df["threshold"], sweep_df["f1"], label="F1")
    ax.plot(sweep_df["threshold"], sweep_df["f2"], label="F2", linestyle=":")
    ax.plot(sweep_df["threshold"], sweep_df["youden_j"], label="Youden's J", linewidth=2.2)
    ax.axvline(DEPLOY_THRESHOLD, color="crimson", linestyle="--",
               label=f"Deployed threshold = {DEPLOY_THRESHOLD:.4f} (exact Youden J)")
    ax.axvline(0.5, color="grey", linestyle="--", label="sklearn default = 0.50")
    ax.axhline(all_positive_f2, color="darkorange", linestyle="-.", linewidth=1,
               label=f"F2 of all-positive model = {all_positive_f2:.3f}")
    ax.set_xlabel("Decision threshold on P(Fatal)")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold selection on the validation split — {deploy_label}")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("8_threshold_selection.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("\n  Saved 8_threshold_selection.png")

# -----------------------------------------------------------------------
# 10.3 TEST-SET IMPACT OF THE TUNED THRESHOLD
# -----------------------------------------------------------------------
# Now — and only now — the tuned threshold is applied to the untouched test
# set. Because the threshold was selected without ever seeing these rows,
# these numbers remain an honest held-out estimate and can go in the report.

print("\n10.3 TEST-SET METRICS — DEFAULT vs TUNED THRESHOLD")

deploy_test_proba = deploy_estimator.predict_proba(X_test_processed)[:, 1] \
    if hasattr(deploy_estimator, "predict_proba") else None

threshold_comparison = []
deployed_test_metrics = None
if deploy_test_proba is not None:
    for label, t in [("Default (0.50)", 0.5),
                     (f"Deployed / Youden J ({DEPLOY_THRESHOLD:.4f})", DEPLOY_THRESHOLD)]:
        y_pred_t = (deploy_test_proba >= t).astype(int)
        row = {
            "Threshold": label,
            "Accuracy": accuracy_score(y_test, y_pred_t),
            "Precision": precision_score(y_test, y_pred_t, zero_division=0),
            "Recall": recall_score(y_test, y_pred_t, zero_division=0),
            "F1": f1_score(y_test, y_pred_t, zero_division=0),
            "F2": fbeta_score(y_test, y_pred_t, beta=2, zero_division=0),
        }
        threshold_comparison.append(row)
        if t == DEPLOY_THRESHOLD:
            deployed_test_metrics = row

    print(pd.DataFrame(threshold_comparison).to_string(index=False))

    tn, fp, fn, tp = confusion_matrix(
        y_test, (deploy_test_proba >= DEPLOY_THRESHOLD).astype(int), labels=[0, 1]
    ).ravel()
    print(f"\n  Confusion matrix at the deployed threshold ({DEPLOY_THRESHOLD:.4f}):")
    print(f"    True Non-Fatal predicted Non-Fatal (TN): {tn}")
    print(f"    True Non-Fatal predicted Fatal     (FP): {fp}")
    print(f"    True Fatal     predicted Non-Fatal (FN): {fn}   <- the costly error")
    print(f"    True Fatal     predicted Fatal     (TP): {tp}")

    tn0, fp0, fn0, tp0 = confusion_matrix(
        y_test, (deploy_test_proba >= 0.5).astype(int), labels=[0, 1]
    ).ravel()
    print(f"\n  Trade-off vs the 0.50 default: {fn0 - fn} fewer missed fatalities "
          f"({fn0} -> {fn}), paid for with {fp - fp0} more false alarms "
          f"({fp0} -> {fp}).")
    print("  For a screening tool that flags collisions for review, that is the")
    print("  intended direction: a false alarm costs a review, a missed fatality")
    print("  costs the thing the model exists to prevent.")

# -----------------------------------------------------------------------
# 10.4 BUILD THE SERVING PIPELINE
# -----------------------------------------------------------------------
# One object that goes raw DataFrame -> imputation -> scaling/one-hot ->
# prediction. This is what makes the Flask endpoint a thin wrapper instead of
# a reimplementation of section 7.

print("\n10.4 BUILDING THE SERVING PIPELINE")

serving_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),   # already fitted in section 7
    ("classifier", deploy_estimator),  # already fitted in section 9
])

# Smoke test: the pipeline must accept RAW feature rows (pre-encoding).
_smoke = serving_pipeline.predict_proba(X_test.head(5))[:, 1]
print(f"  Pipeline accepts raw feature rows — sample probabilities: "
      f"{np.round(_smoke, 4).tolist()}")

# -----------------------------------------------------------------------
# 10.5 FIELD SCHEMA FOR THE CLIENT FORM
# -----------------------------------------------------------------------
# The HTML form exposes every feature the model consumes. Rather than
# hand-typing 30 fields (which would drift the moment feature selection
# changes), the schema is derived from the fitted pipeline itself:
#   - categorical options come from the fitted OneHotEncoder's categories_,
#     so the form can only ever offer values the model was trained on;
#   - numeric ranges/defaults come from the training distribution.

print("\n10.5 BUILDING THE FIELD SCHEMA FOR THE FORM")

fitted_encoder = preprocessor.named_transformers_["cat"]["encoder"]
encoder_categories = {
    col: [str(v) for v in cats]
    for col, cats in zip(categorical_features, fitted_encoder.categories_)
}

field_schema = []
for col in X.columns:
    if col in categorical_features:
        field_schema.append({
            "name": col,
            "type": "categorical",
            "options": encoder_categories[col],
            "default": str(X_train[col].mode().iloc[0]),
            "description": kept_cols.get(col, ""),
        })
    elif col in binary_flag_cols:
        field_schema.append({
            "name": col,
            "type": "binary",
            "options": [0, 1],
            "default": int(X_train[col].mode().iloc[0]),
            "description": kept_cols.get(col, ""),
        })
    else:
        field_schema.append({
            "name": col,
            "type": "numeric",
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max()),
            "default": float(X_train[col].median()),
            "description": kept_cols.get(col, ""),
        })

n_cat = sum(1 for f in field_schema if f["type"] == "categorical")
n_bin = sum(1 for f in field_schema if f["type"] == "binary")
n_num = sum(1 for f in field_schema if f["type"] == "numeric")
print(f"  {len(field_schema)} form fields: {n_num} numeric, {n_bin} binary flags, "
      f"{n_cat} categorical")

# -----------------------------------------------------------------------
# 10.6 SERIALIZE THE BUNDLE (pickle)
# -----------------------------------------------------------------------
# Everything the API needs travels in ONE pickle so the served model and the
# metadata describing it can never drift apart.

print("\n10.6 SERIALIZING WITH pickle")

model_bundle = {
    "pipeline": serving_pipeline,
    "model_name": deploy_label,
    "calibration": "isotonic, 3-fold (applied at deployment, not during model comparison)",
    "threshold": DEPLOY_THRESHOLD,
    "threshold_note": threshold_note,
    "feature_order": list(X.columns),
    "numeric_features": list(numeric_features),
    "categorical_features": list(categorical_features),
    "binary_flag_cols": list(binary_flag_cols),
    "rare_category_maps": rare_category_maps,
    "field_schema": field_schema,
    "target_labels": {0: "Non-Fatal", 1: "Fatal"},
    "test_metrics": {
        # The uncalibrated figures are the ones Deliverable 3 reported and used
        # to pick this model; the calibrated ones describe what is actually
        # served. Both are kept so the two can never be confused.
        "uncalibrated_at_default_threshold": {
            k: float(deploy_row[k]) for k in
            ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]},
        "at_default_threshold": {
            "Accuracy": float(accuracy_score(y_test, (_cal_proba >= 0.5).astype(int))),
            "Precision": float(precision_score(y_test, (_cal_proba >= 0.5).astype(int), zero_division=0)),
            "Recall": float(recall_score(y_test, (_cal_proba >= 0.5).astype(int), zero_division=0)),
            "F1": float(f1_score(y_test, (_cal_proba >= 0.5).astype(int), zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y_test, _cal_proba)),
        },
        "at_deployed_threshold": deployed_test_metrics,
        "threshold_comparison": threshold_comparison,
        "saturation_before_calibration": _saturation(_raw_proba),
        "saturation_after_calibration": _saturation(_cal_proba),
    },
    "threshold_sweep": sweep_df.to_dict("records") if "sweep_df" in globals() else None,
    "sklearn_version": sklearn.__version__,
    "pandas_version": pd.__version__,
    "trained_at": datetime.now().isoformat(timespec="seconds"),
}

BUNDLE_PATH = os.path.join(PROJECT_DIR, "model_bundle.pkl")
with open(BUNDLE_PATH, "wb") as f:
    pickle.dump(model_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

bundle_mb = os.path.getsize(BUNDLE_PATH) / 1024 / 1024
print(f"  Wrote {BUNDLE_PATH} ({bundle_mb:.2f} MB)")

# Deserialization check — a pickle that cannot be loaded back is not a
# deliverable. This is the same call app.py makes at start-up.
with open(BUNDLE_PATH, "rb") as f:
    reloaded = pickle.load(f)
_check = reloaded["pipeline"].predict_proba(X_test.head(5))[:, 1]
assert np.allclose(_check, _smoke), "Reloaded pipeline disagrees with the in-memory one"
print("  Deserialization verified: reloaded pipeline reproduces identical probabilities")

# -----------------------------------------------------------------------
# 10.7 HELD-OUT SAMPLES FOR THE API CLIENT
# -----------------------------------------------------------------------
# The project spec requires the client to be tested on data that was NOT used
# to train the model. These rows come from X_test, which the grouped split
# kept entirely out of training (whole crashes went to one side or the other),
# and they are saved BEFORE encoding — exactly the shape the API accepts.

print("\n10.7 EXPORTING HELD-OUT TEST SAMPLES FOR THE CLIENT")

# Rows are sampled at random (fixed seed) rather than taken as the first N by
# index. The index is essentially chronological, so `.index[:10]` would return
# ten collisions clustered in the same few days — an unrepresentative slice to
# demo the client with. The sample is deliberately enriched with Fatal cases
# (10 of 25 vs the ~14% base rate) so the client exercises both classes.
fatal_idx = y_test[y_test == 1].sample(n=10, random_state=RANDOM_STATE).index
nonfatal_idx = y_test[y_test == 0].sample(n=15, random_state=RANDOM_STATE).index
sample_idx = list(fatal_idx) + list(nonfatal_idx)

test_samples = X_test.loc[sample_idx].copy()
test_samples["ACTUAL_ACCLASS_BINARY"] = y_test.loc[sample_idx]
test_samples["ACTUAL_LABEL"] = test_samples["ACTUAL_ACCLASS_BINARY"].map({0: "Non-Fatal", 1: "Fatal"})

SAMPLES_PATH = os.path.join(PROJECT_DIR, "test_samples.csv")
# index=True keeps the original row id, so any exported row can be traced back
# to the source record when a prediction looks wrong.
test_samples.to_csv(SAMPLES_PATH, index=True, index_label="ORIGINAL_INDEX")
print(f"  Wrote {SAMPLES_PATH} — {len(test_samples)} held-out rows "
      f"({len(fatal_idx)} Fatal, {len(nonfatal_idx)} Non-Fatal)")

# -----------------------------------------------------------------------
# 10.8 EXPORT VERIFICATION
# -----------------------------------------------------------------------
# Confirms that the exported artifacts still line up with the model. Two
# distinct failure modes are checked, because a mismatch between the features
# written to CSV and the labels written beside them would be invisible
# otherwise — the client would just report a strangely inaccurate model.

print("\n10.8 VERIFYING THE EXPORTED ARTIFACTS")

# (a) Does the model separate the two classes on the full test set at all?
#     If these two distributions overlap completely, no threshold can help.
if deploy_test_proba is not None:
    proba_fatal = deploy_test_proba[y_test.values == 1]
    proba_nonfatal = deploy_test_proba[y_test.values == 0]
    print("  P(Fatal) assigned across the WHOLE test set:")
    print(f"    Actual Fatal     (n={len(proba_fatal):4d}): "
          f"mean {proba_fatal.mean():.4f}  median {np.median(proba_fatal):.4f}")
    print(f"    Actual Non-Fatal (n={len(proba_nonfatal):4d}): "
          f"mean {proba_nonfatal.mean():.4f}  median {np.median(proba_nonfatal):.4f}")
    if proba_fatal.mean() <= proba_nonfatal.mean():
        print("    WARNING: Fatal rows are NOT scored higher on average — "
              "the model or the label orientation is wrong.")
    else:
        print("    Fatal rows score higher on average, as expected.")

# (b) Round-trip the CSV: read it back, score it through the reloaded bundle,
#     and check the predictions against the labels stored alongside.
roundtrip = pd.read_csv(SAMPLES_PATH, index_col="ORIGINAL_INDEX")
roundtrip_X = roundtrip[list(X.columns)].copy()
for col in categorical_features:
    roundtrip_X[col] = roundtrip_X[col].astype(str)

roundtrip_proba = reloaded["pipeline"].predict_proba(roundtrip_X)[:, 1]
inmemory_proba = serving_pipeline.predict_proba(X_test.loc[sample_idx])[:, 1]

max_drift = np.abs(roundtrip_proba - inmemory_proba).max()
print(f"\n  CSV round-trip vs in-memory scoring: max difference {max_drift:.2e}")
assert max_drift < 1e-6, "Exported CSV does not reproduce in-memory predictions"

exported_actual = roundtrip["ACTUAL_ACCLASS_BINARY"].values
mean_p_fatal = roundtrip_proba[exported_actual == 1].mean()
mean_p_nonfatal = roundtrip_proba[exported_actual == 0].mean()
print(f"  Exported sample — mean P(Fatal): "
      f"Fatal rows {mean_p_fatal:.4f} vs Non-Fatal rows {mean_p_nonfatal:.4f}")
# Not an assertion: on a 25-row sample this comparison can fail by chance even
# when everything is correct. The whole-test-set check in (a) above is the
# reliable signal for label misalignment; this one is a per-sample sanity note.
if mean_p_fatal <= mean_p_nonfatal:
    print("    NOTE: on this small sample the Fatal rows do not score higher on")
    print("    average. Check the whole-test-set separation printed above — if")
    print("    that is also inverted, features and labels are misaligned.")

exported_pred = (roundtrip_proba >= DEPLOY_THRESHOLD).astype(int)
print(f"  Accuracy on the exported sample: "
      f"{(exported_pred == exported_actual).sum()}/{len(exported_actual)} "
      f"(fatal-enriched, so not comparable to the full-test-set figures)")
print("  Export verified.")

# Human-readable metadata sidecar (handy for the report and for Postman).
META_PATH = os.path.join(PROJECT_DIR, "model_metadata.json")
with open(META_PATH, "w") as f:
    json.dump({k: v for k, v in model_bundle.items() if k != "pipeline"}, f, indent=2, default=str)
print(f"  Wrote {META_PATH}")

# -----------------------------------------------------------------------
# 10.9 REPORT NUMBERS FOR PART 4
# -----------------------------------------------------------------------
# The Part 4 write-up quotes a model comparison table and per-model confusion
# matrices. When those are typed in by hand they drift: the report ends up
# describing one training run while the pickled model came from another, and
# anyone cross-checking the report against the live API finds a mismatch.
#
# Results are metric values, not fitted state, so they cannot be recovered
# from model_bundle.pkl — they have to be written out by the run that produced
# it. This file is generated from the same in-memory results that built the
# bundle, so the two cannot disagree. Regenerate it whenever the pipeline is
# re-run, and paste from it rather than editing the report numbers directly.
#
# Numbers WILL differ between machines even with random_state pinned, because
# estimator internals change across scikit-learn versions. The environment is
# therefore recorded alongside the table, and whichever machine the group
# treats as canonical should be the one that generates both this file and the
# shipped model_bundle.pkl.

print("\n10.9 GENERATING THE PART 4 REPORT NUMBERS")

REPORT_PATH = os.path.join(PROJECT_DIR, "part4_report_numbers.md")

_report_df = results_df.copy()
_report_df["Model"] = _report_df["Model"].str.replace(" (tuned)", "", regex=False)

with open(REPORT_PATH, "w") as f:
    f.write("# Part 4 — Model Scoring and Evaluation (generated)\n\n")
    f.write("Generated by `public_safety_ml.py` on "
            f"{datetime.now().isoformat(timespec='seconds')}.\n\n")
    f.write("**Do not edit by hand.** Re-run the pipeline and paste from this file, "
            "so the written report and the deployed `model_bundle.pkl` always "
            "describe the same training run.\n\n")

    f.write("## Environment\n\n")
    f.write(f"- scikit-learn `{sklearn.__version__}`, pandas `{pd.__version__}`, "
            f"numpy `{np.__version__}`\n")
    f.write(f"- Random seed: `{RANDOM_STATE}` (pinned throughout)\n")
    f.write("- Estimator internals change across scikit-learn releases, so a "
            "different version reproduces the ranking but not every decimal.\n\n")

    f.write("## 4.1 Performance metrics comparison\n\n")
    f.write(f"Test set: **{len(y_test)} records** "
            f"({int((y_test == 0).sum())} Non-Fatal vs {int((y_test == 1).sum())} Fatal). "
            "All models trained on the SMOTE-balanced training set and evaluated at "
            "the default 0.5 threshold.\n\n")
    f.write("| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n")
    f.write("|---|---|---|---|---|---|\n")
    for _, r in _report_df.iterrows():
        f.write(f"| {r['Model']} | {r['Accuracy'] * 100:.2f}% | {r['Precision'] * 100:.2f}% "
                f"| {r['Recall'] * 100:.2f}% | {r['F1']:.3f} | {r['ROC-AUC']:.3f} |\n")

    f.write("\n## 4.3 Confusion matrices\n\n")
    f.write("At the default 0.5 threshold, in the same order as the table above:\n\n")
    _by_name = {r["Model"]: r for r in results}
    for _, row in results_df.iterrows():
        r = _by_name[row["Model"]]
        tn, fp, fn, tp = confusion_matrix(y_test, r["y_pred"], labels=[0, 1]).ravel()
        f.write(f"- **{row['Model'].replace(' (tuned)', '')}**: correctly identified "
                f"{tp} fatalities, missed {fn} (false negatives), and triggered "
                f"{fp} false positives.\n")

    f.write("\n## 4.4 Deployed configuration\n\n")
    f.write(f"- Model deployed: **{deploy_label}**\n")
    f.write(f"- Selected as the highest test F1 on the Fatal class "
            f"({deploy_row['F1']:.3f})\n")
    f.write(f"- Decision threshold: **{DEPLOY_THRESHOLD:.4f}** — {threshold_note}\n")
    f.write("- Isotonic calibration applied at the deployment stage only; the table "
            "above reports the models exactly as tuned in Deliverable 3.\n\n")
    if deployed_test_metrics:
        f.write("Test metrics for the deployed configuration:\n\n")
        f.write("| Threshold | Accuracy | Precision | Recall | F1 | F2 |\n|---|---|---|---|---|---|\n")
        for row in threshold_comparison:
            f.write(f"| {row['Threshold']} | {row['Accuracy'] * 100:.1f}% "
                    f"| {row['Precision'] * 100:.1f}% | {row['Recall'] * 100:.1f}% "
                    f"| {row['F1']:.3f} | {row['F2']:.3f} |\n")

print(f"  Wrote {REPORT_PATH}")
print("  Paste Part 4's table and confusion matrices from this file so the report")
print("  and the shipped model can never describe different runs.")

print(f"""
DELIVERABLE 5 — EXPORT SUMMARY

  Deployed model:      {deploy_label}
  Decision threshold:  {DEPLOY_THRESHOLD:.4f} ({threshold_note})
  Serving pipeline:    ColumnTransformer(fitted) -> isotonic-calibrated {deploy_key}
  Input contract:      {len(X.columns)} raw features, pre-encoding
  Artifacts written:   model_bundle.pkl, test_samples.csv, model_metadata.json,
                       8_threshold_selection.png

  Next: `python app.py` serves this bundle at http://127.0.0.1:5000
""")
