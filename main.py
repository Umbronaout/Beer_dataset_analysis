import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsRegressor
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import time
from textblob import TextBlob

np.random.seed(42)
# Transforms the "review_text" cloumn into polarity using TextBlob
use_polarity = False
# Converts column "beer_style" - Clusters styles and uses one-hot encoding
use_beer_styles = False

def round_half(x):
    return np.clip(np.round(x * 2) / 2, a_min=0, a_max=5)

def rounded_rmse(y_true, y_pred):
    y_pred_rounded = round_half(y_pred)
    mse = mean_squared_error(y_true, y_pred_rounded)
    return np.sqrt(mse)

custom_scorer = make_scorer(rounded_rmse, greater_is_better=False)

def get_polarity(text):
    return TextBlob(str(text)).sentiment.polarity

def log(text):
    print(text)
    with open("Log.txt", "a") as l:
        l.write(text + "\n")


# Load data
df = pd.read_csv("BeerProject.csv", encoding='latin-1')

drop_cols = [
    "beer_beerId", "beer_brewerId", "beer_name", "review_profileName", "review_time"
]
df.drop(columns=drop_cols, inplace=True, errors="ignore")

# Process text column if specified
if use_polarity:
    df['polarity'] = df['review_text'].fillna('').apply(get_polarity)

df.drop(columns=['review_text'], inplace=True, errors="ignore")

# Drop rows with NaNs
df.dropna(inplace=True)
df = df[df['review_overall'] != 0]

# Print basic info
print("Data Information:")
df.info()
print("\nSample of data:")
print(df.head())

# Basic statistics
print("\nDistribution of ratings:")
print(df["review_overall"].value_counts().sort_index())
plt.figure(figsize=(10, 6))
plt.hist(df["review_overall"], bins=10, edgecolor='black')
plt.title('Distribution of Beer Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('rating_distribution.png')

### Cluster beer styles ###
# unique beer styles
beer_styles = df["beer_style"].unique()

# TF-IDF
vectorizer = TfidfVectorizer()
X_styles = vectorizer.fit_transform(beer_styles).toarray()

# linkage matrix
cut_off = 1.48
linkage_matrix = linkage(X_styles, method='ward')

# dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, labels=beer_styles, leaf_rotation=90, leaf_font_size=10, color_threshold=cut_off)
plt.title('Hierarchical Clustering of Beer Styles')
plt.xlabel('Beer Styles')
plt.ylabel('Distance')
plt.axhline(y=cut_off, c='red', ls='--', lw=1)
plt.tight_layout()
plt.savefig('beer_style_clusters.png')

clusters = fcluster(linkage_matrix, t=cut_off, criterion='distance')

# Create pd Datafrome for merging
clustered_styles = pd.DataFrame({
    'beer_style': beer_styles,
    'style_cluster': clusters
})

if use_beer_styles:
    # Merge
    df = df.merge(clustered_styles, on='beer_style', how='left')
    entry_counts = df['style_cluster'].value_counts().sort_index()

    # Plot
    plt.figure(figsize=(10, 6))
    entry_counts.plot(kind='bar')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Entries')
    plt.title('Number of Entries in Each Cluster')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('entries_per_cluster.png')
    plt.show()

    # log info
    num_clusters = len(np.unique(clusters))
    log(f"\nNumber of beer style clusters: {num_clusters}")
    cluster_counts = clustered_styles.groupby('style_cluster').size().sort_values(ascending=False)
    print("\nCluster sizes:")
    print(cluster_counts)

    print("\nbeer styles in each cluster:")
    for cluster_id in np.unique(clusters):
        styles_in_cluster = clustered_styles[clustered_styles['style_cluster'] == cluster_id]['beer_style'].values
        log(f"Cluster {cluster_id} ({len(styles_in_cluster)} styles): {', '.join(styles_in_cluster)}")

# target and features
y = df["review_overall"]
X = df.drop(columns=["review_overall", "beer_style"])  # Drop original beer_style as we're using clusters

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.02, train_size=0.08, random_state=42
)

# Identify categorical and numerical columns
if use_beer_styles:
    cat_cols = ["style_cluster"]
    num_cols = [col for col in X.columns if col not in cat_cols and X[col].dtype in ["int64", "float64"]]# 
else:
    num_cols = [col for col in X.columns if X[col].dtype in ["int64", "float64"]]
log(f"\nNumerical columns:{num_cols}")
if use_polarity:
    log(f"Categorical columns:{cat_cols}")

# Preprocess
if use_beer_styles:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ], verbose_feature_names_out=False)
else:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols)
    ], verbose_feature_names_out=False)   

# Models
models = {
    "MLPRegressor": MLPRegressor(max_iter=10000, early_stopping=True, random_state=42),
    "KNN": KNeighborsRegressor(),
    "RandomForest": RandomForestRegressor(random_state=42),
}

# Param grids
param_grids = {
    "MLPRegressor": {
        "model__hidden_layer_sizes": [(layer * 3,) for layer in range(3, 9)] + [(layer1 * 3, layer2 * 3) for layer1 in range(3, 9) for layer2 in range(3, 9)],
        "model__alpha": [0.5, 1, 2],
        "model__learning_rate_init": [0.001, 0.01],
        "model__solver": ["adam"],
        "model__activation": ["relu"],
        "model__learning_rate": ["constant"]
    },

    "KNN": {
        "model__n_neighbors": [step * 2 for step in range(5, 26)],
        "model__weights": ["uniform", "distance"],
        "model__algorithm": ["auto"],
        "model__leaf_size": [30, 50]
    },

    "RandomForest": {
        "model__n_estimators": [20 * n for n in range(2, 11)],
        "model__max_depth": [5, 10, 15, 20, None],
        "model__min_samples_split": [2, 3, 4],
        "model__max_features": ["sqrt"]
    }
}

# Function to train and evaluate a model
def train_and_evaluate_model(model_name, model, param_grid, X_train, y_train, X_test, y_test):
    log(f"\n{'-'*50}")
    # Train
    log(f"Training {model_name}...")
    start_time = time.time()
    
    steps = []
    steps.append(("preprocessing", preprocessor))
    
    steps.append(("model", model))
    
    pipeline = Pipeline(steps)
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring=custom_scorer,
        verbose=10,
        n_jobs=4
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_parameters = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Test
    y_test_pred = best_model.predict(X_test)
    mse = rounded_rmse(y_test, y_test_pred)
    r2 = r2_score(y_test, round_half(y_test_pred))
    
    duration = time.time() - start_time
    
    log(f"Training completed in {duration:.2f} seconds")
    log(f"Best CV RMSE: {-best_score:.4f}")  # Note: Custom scorer is negative for GridSearchCV
    log(f"Test RMSE: {mse:.4f}")
    log(f"Test R² Score: {r2:.4f}")
    log(f"Best Parameters: {best_parameters}")
    
    # Plot predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, round_half(y_test_pred), alpha=0.3)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title(f'{model_name} - Predicted vs Actual Ratings')
    plt.savefig(f'{model_name}_predictions.png')
    
    return {
        'model_name': model_name,
        'best_model': best_model,
        'best_params': best_parameters,
        'cv_rmse': -best_score,
        'test_rmse': mse,
        'r2': r2,
        'training_time': duration
    }

# Train all
results = []
for model_name, model in models.items():
    result = train_and_evaluate_model(
        model_name=model_name,
        model=model,
        param_grid=param_grids[model_name],
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    results.append(result)

# results
log("\n" + "="*50)
log("MODEL COMPARISON")
log("="*50)
log(f"{'Model':<15} {'CV RMSE':<10} {'Test RMSE':<10} {'R²':<10} {'Time (s)':<10}")
log("-"*55)
for result in sorted(results, key=lambda x: x['test_rmse']):
    log(f"{result['model_name']:<15} {result['cv_rmse']:.4f}{' ':<10} {result['test_rmse']:.4f}{' ':<10} {result['r2']:.4f}{' ':<10} {result['training_time']:.2f}")

# Best model
best_model_result = min(results, key=lambda x: x['test_rmse'])
log(f"\nBest model: {best_model_result['model_name']} with Test RMSE of {best_model_result['test_rmse']:.4f}")

best_model = best_model_result['best_model']
y_test_pred = best_model.predict(X_test)
y_test_pred_rounded = round_half(y_test_pred)

log("\nAnalysis complete. Check the generated plots.")