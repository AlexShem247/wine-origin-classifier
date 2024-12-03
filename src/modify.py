import pandas as pd


def process_csv(input_file, output_file, top_n):
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Define the columns to skip (these will not be processed)
    columns_to_skip = ['id', 'country', 'description', 'points', 'price', 'variety']

    # Print the number of columns before processing
    total_columns_before = len(df.columns)
    print(f"Total columns in the original dataset: {total_columns_before}")

    # Dictionary to store sums of columns
    column_sums = {}

    # Iterate through each column and calculate the sum for "other columns"
    for column in df.columns:
        if column not in columns_to_skip:
            # Convert the column to numeric (if it's not already)
            df[column] = pd.to_numeric(df[column], errors='coerce')

            # Calculate the sum of the column
            column_sums[column] = df[column].sum()

    # Sort the columns based on their sum in descending order and get the top N
    top_columns = sorted(column_sums, key=column_sums.get, reverse=True)[:top_n]

    # Keep only the top N columns, along with the skipped columns
    columns_to_keep = columns_to_skip + top_columns
    df = df[columns_to_keep]

    # Print the number of columns kept
    total_columns_after = len(df.columns)
    print(f"Total columns kept after processing: {total_columns_after}")

    # Save the processed DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

    # Print the sum of the processed columns
    print("Processed file saved as", output_file)
    print(f"\nTop {top_n} most important columns based on the sum of values:")
    for column in top_columns:
        print(f"{column}: {column_sums[column]}")


if __name__ == "__main__":
    input_file = '../output/wine_quality_features+_1000.csv'  # Replace with your actual input file path
    output_file = '../output/new_modified.csv'  # Replace with your desired output file path

    process_csv(input_file, output_file, 500)
