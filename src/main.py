import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree


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

        # Drop unused column and separate features and target
        self.df.drop(columns=["description"], inplace=True)
        self.X = self.df[["price", "points"] + [col for col in self.df if col.startswith("variety_")]]

        # Label encode the target variable and create country lookup dictionary
        label_encoder = LabelEncoder()
        self.y = label_encoder.fit_transform(self.df["country"])
        self.country_lookup = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

        print("Data preprocessing completed.")

        # Split data for training and testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.1,
                                                                                random_state=42)
        return self.X_train, self.X_test

    def hyperparameter_tuning(self):
        """Tune hyperparameters for max_depth and n_estimators."""
        # Define the parameter grid for max_depth and n_estimators
        param_grid = {
            "n_estimators": [100, 200, 300, 500, 700],
            "max_depth": [10, 20, 30, 40, 50]
        }

        best_score = 0
        best_params = {}

        # Loop over possible parameter combinations
        for n_estimators in param_grid["n_estimators"]:
            for max_depth in param_grid["max_depth"]:
                # Initialise the model with current hyperparameters
                clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

                # Split the data into train and validation sets (using the train set only)
                X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.y_train, test_size=0.1,
                                                                  random_state=42)

                # Train the model
                clf.fit(X_train, y_train)

                # Evaluate on the validation set
                score = clf.score(X_val, y_val)

                # Keep track of the best score and parameters
                if score > best_score:
                    best_score = score
                    best_params = {"n_estimators": n_estimators, "max_depth": max_depth}

        print(f"Best Hyperparameters: {best_params}")
        print(f"Best Validation Accuracy: {best_score:.2f}")

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

    def visualise_tree(self, tree_index=0):
        """Visualise one tree from the random forest (by default the first tree)."""
        plt.figure(figsize=(12, 8))  # Adjust size for readability
        plot_tree(self.random_forest.estimators_[tree_index],
                  feature_names=self.X.columns, class_names=[str(c) for c in self.country_lookup.keys()],
                  filled=True)
        plt.show()


if __name__ == "__main__":
    wine_data_source = "data/wine_quality_1000.csv"

    wine_data = WineOriginClassifier(wine_data_source)

    wine_data.load_data()
    X, y = wine_data.preprocess_data()

    # Hyperparameter tuning to improve the model
    params = wine_data.hyperparameter_tuning()

    # Train the model again with the best hyperparameters
    wine_data.train_random_forest(*params)

    # Visualise one tree from the forest
    wine_data.visualise_tree()
