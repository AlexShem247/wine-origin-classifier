import pandas as pd


def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)

    # List of columns to ignore
    ignore_columns = ['id', 'description', 'points', 'price', 'variety']

    # Loop through all columns except the ones to ignore
    for column in df.columns:
        if column in ignore_columns:
            continue

        # If the column contains a flavour-related feature (flavour or favour)
        if 'flavour' in column.lower() or 'favour' in column.lower():  # Check both 'flavour' and 'favour'
            flavour_type = column.lower().replace('flavour', '').replace('favour', '').strip()
            df[column].fillna(f'not-{flavour_type}', inplace=True)

        # For other columns, fill with the mode (most frequent value)
        else:
            mode_value = df[column].mode()[0]  # Get the most frequent value
            df[column].fillna(mode_value, inplace=True)

    # Save the updated dataframe to the output file
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    input_file = '../output/wine_quality_more_features_500.csv'
    output_file = '../output/new_modified.csv'

    process_csv(input_file, output_file)
