import graphviz
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
from textblob import TextBlob


class WineOriginClassifier:
    def __init__(self, data_path):
        # Initialise with the path to the CSV file
        self.data_path = data_path
        self.df = None
        self.country_lookup = None

        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.random_forest = None

    def load_data(self):
        """Load the data from the CSV file."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully with {len(self.df)} rows.")
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def preprocess_data(self):
        """Preprocess the data: encodes categorical features and splits data for training and testing."""
        if self.df is None:
            print("Data not loaded. Please load the data first.")
            return None

        # Apply one-hot encoding
        self.df = pd.get_dummies(self.df, columns=["variety"], drop_first=True)

        # Clean column names and convert boolean columns to 0 and 1
        self.df.columns = [col.lower().replace(" ", "_").encode("ascii", "ignore").decode("ascii")
                           for col in self.df.columns]
        self.df = self.df.map(lambda x: 1 if x is True else (0 if x is False else x))

        # Drop the "description" column for sentiment analysis
        descriptions = self.df.pop("description")

        # Apply sentiment analysis to the descriptions
        self.df["sentiment_polarity"] = descriptions.apply(self.get_sentiment_polarity)
        self.df["sentiment_subjectivity"] = descriptions.apply(self.get_sentiment_subjectivity)

        # Apply TF-IDF Vectorization on "description" column
        tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        X_desc = tfidf_vectorizer.fit_transform(descriptions)

        # Get the feature names and create new column names for descriptions (e.g., "desc_{word}")
        desc_columns = [f"desc_{word}" for word in tfidf_vectorizer.get_feature_names_out()]

        # Convert the sparse matrix to a dense DataFrame and set the column names
        X_desc_df = pd.DataFrame(X_desc.toarray(), columns=desc_columns)

        # Concatenate the new description features with the other features (price, points, variety)
        self.df = pd.concat([self.df, X_desc_df], axis=1)

        # Define features (X) and target (y)
        self.X = self.df[["price", "points"] + [col for col in self.df if col.startswith("variety_")] + desc_columns]

        # Label encode the target variable and create country lookup dictionary
        label_encoder = LabelEncoder()
        self.y = label_encoder.fit_transform(self.df["country"])
        self.country_lookup = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

        print("Data preprocessing completed.")

        # Split data for training and testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.1,
                                                                                random_state=42)
        return self.X_train, self.X_test

    def get_sentiment_polarity(self, text):
        """Returns the sentiment polarity of the given text."""
        return TextBlob(text).sentiment.polarity  # Ranges from -1 (negative) to 1 (positive)

    def get_sentiment_subjectivity(self, text):
        """Returns the subjectivity score of the given text."""
        return TextBlob(text).sentiment.subjectivity  # Ranges from 0 (objective) to 1 (subjective)

    def hyperparameter_tuning(self, skip=False):
        """Tune hyperparameters for max_depth and n_estimators."""
        param_grid = {
            "n_estimators": [100, 200, 300, 500, 700],
            "max_depth": [10, 20, 30, 40, 50]
        }

        if skip:
            return param_grid["n_estimators"][4], param_grid["max_depth"][3]

        clf = RandomForestClassifier(random_state=42)

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring="f1_weighted", n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        # Get the best parameters and score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print(f"Best Hyperparameters: {best_params}")
        print(f"Best Cross-Validation Accuracy: {best_score:.2f}")

        return best_params["n_estimators"], best_params["max_depth"]

    def train_random_forest(self, n_estimators=100, max_depth=None):
        """Train Random Forest with specified hyperparameters."""
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        # Train the model on the whole training data
        clf.fit(self.X_train, self.y_train)

        # Save the trained model as a class field
        self.random_forest = clf

        # Test the model on the final test set
        test_score = clf.score(self.X_test, self.y_test)
        print(f"Test Accuracy: {test_score:.2f}")

        return clf

    def visualise_tree(self, tree_index=0, max_depth=3):
        """Visualise one tree from the random forest (by default the first tree)."""
        dot_data = export_graphviz(self.random_forest.estimators_[tree_index],
                                   feature_names=self.X.columns,
                                   class_names=[str(c) for c in self.country_lookup.keys()],
                                   filled=True, rounded=True,
                                   special_characters=True,
                                   max_depth=max_depth)

        # Create the Graphviz source object
        graph = graphviz.Source(dot_data)

        # Render and display the tree
        graph.render(filename="random_forest_tree", view=True)


if __name__ == "__main__":
    wine_data_source = "data/wine_quality_1000.csv"

    wine_data = WineOriginClassifier(wine_data_source)

    wine_data.load_data()
    X, y = wine_data.preprocess_data()

    # Hyperparameter tuning to improve the model
    params = wine_data.hyperparameter_tuning(skip=True)

    # Train the model again with the best hyperparameters
    wine_data.train_random_forest(*params)

    # Visualise one tree from the forest
    wine_data.visualise_tree()
