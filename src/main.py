import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


class WineOriginClassifier:
    def __init__(self, data_path):
        # Initialize with the path to the CSV file
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.country_lookup = None

    def load_data(self):
        """Load the data from the CSV file."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully with {len(self.df)} rows.")
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def preprocess_data(self):
        """Preprocess the data: handle missing values and encode categorical features."""
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
        return self.X, self.y

    def show_data_info(self):
        """Print out basic information about the data."""
        if self.df is not None:
            print(self.df.info())
            print(self.df.head())

    def train_decision_tree(self, max_depth=None):
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Initialize and train the decision tree classifier
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)

        # Predict on the test set
        y_pred = clf.predict(X_test)

        # Print metrics
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        print("Weighted F1-Score:", weighted_f1)

        # Save the trained model as a class field
        self.decision_tree = clf

        return clf

    def visualize_tree(self):
        plt.figure(figsize=(12, 8))  # Adjust size for readability
        plot_tree(self.decision_tree, feature_names=self.X.columns, class_names=[str(c) for c in
                                                                                 self.country_lookup.keys()],
                  filled=True)
        plt.show()


if __name__ == "__main__":
    wine_data_source = "data/wine_quality_1000.csv"

    wine_data = WineOriginClassifier(wine_data_source)

    wine_data.load_data()
    X, y = wine_data.preprocess_data()
    wine_data.train_decision_tree(max_depth=10)
    wine_data.visualize_tree()
