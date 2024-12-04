import pandas as pd

def process_csv(input_file, other_file, output_file):
    # Read both CSV files
    df1 = pd.read_csv(input_file)
    df2 = pd.read_csv(other_file)

    # Merge both DataFrames on the 'id' column
    merged_df = pd.merge(df1, df2[['id', 'price', 'points', 'variety']], on='id', how='left')

    # Save the merged DataFrame to a new CSV
    merged_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = '../output/new_modified.csv'
    other_file = '../data/wine_quality_1000.csv'
    output_file = '../output/modified.csv'

    process_csv(input_file, other_file, output_file)
