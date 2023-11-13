def header(str):
    return f'\n{"="*50}\n{str}\n{"="*50}\n'

def df_to_markdown(dataframe, file_path):
    """
    Convert a Pandas DataFrame to a 
    own table and export it to a file.

    Parameters:
    - dataframe (pd.DataFrame): The Pandas DataFrame to convert.
    - file_path (str): The file path for exporting the Markdown table.
    """
    # Convert DataFrame to Markdown
    markdown_table = dataframe.to_markdown(index=False)

    # Export Markdown table to a file
    with open(file_path, 'w') as file:
        file.write(markdown_table)

# Example Usage:
# Assuming you have a DataFrame called 'my_dataframe'
# and you want to export it to a file named 'output_table.md'
# df_to_markdown(my_dataframe, 'output_table.md')