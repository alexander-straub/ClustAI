"""
get_labeled_data.py

Script to extract all the events, clusters and potentially
combined answers from previous .xlsx files.
=> labeled data

get_labeled_data_from_cleaned_dataset is the only relevant function here
since i have created a .xlsx file with all the relevant Cluster-Note pairs.

Author: Alexander Straub
"""

from pathlib import Path
import pandas as pd

def get_labeled_data(root_folder):
    """
    Function that iterates through a root folder and extracts all the events and their corresponding clusters
    from the .xlsx files in the "03_Trigger Logs" subfolders (KPI_Format_Converters are excluded).
    
    Input:
        - root_folder: (str)
            Path to the root folder

    Output:
        - labeled_data: (list of tuples)
            List of tuples with cluster and note values
    """
    root_path = Path(root_folder)
    sheet_name = "Triggerliste"
    note_column = "Note"
    cluster_column = "Cluster"

    # List of .xlsx filenames (without path) to exclude
    excluded_files = {}

    column_cluster_values = []
    column_note_values = []

    # Iterate through year folders (2025, 2026, ...)
    for year_folder in root_path.iterdir():
        if not year_folder.is_dir():
            continue
        # Iterate through first-level subfolders inside each year folder
        for subfolder in year_folder.iterdir():
            if not subfolder.is_dir():
                continue

            trigger_logs_path = subfolder / "03_Trigger Logs"

            if trigger_logs_path.exists() and trigger_logs_path.is_dir():

                # Iterate through xlsx files
                for xlsx_file in trigger_logs_path.glob("*.xlsx"):

                    # Skip files whose name contains any of the excluded substrings
                    if any(excluded in xlsx_file.name for excluded in excluded_files):
                        continue

                    # Read the Excel file
                    df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
                    # Remove rows where either Cluster or Note is empty
                    df = df.dropna(subset=[cluster_column, note_column])
                    
                    # Append lists with clusters and events
                    temp_cluster_list = df[cluster_column].dropna().astype(str).tolist()
                    temp_event_list = df[note_column].dropna().astype(str).tolist()
                    column_cluster_values.extend(temp_cluster_list)
                    column_note_values.extend(temp_event_list)

                    # Clear the lists for next .xlsx file
                    temp_cluster_list.clear()
                    temp_event_list.clear()

    # Create a list of tuples
    labeled_data = list(zip(column_cluster_values, column_note_values))
    # Sort dictionary alphabetically according to the clusters
    labeled_data_sorted = sorted(labeled_data, key=lambda x: (x[0], x[1]))
    # Remove entries with cluster 'ignore'
    labeled_data_sorted = [item for item in labeled_data_sorted if item[0].lower() != 'ignore']
    return labeled_data_sorted


def get_labeled_data_with_comb_answer(root_folder):
    """
    Function that iterates through a root folder and extracts all the events and their corresponding clusters
    and combined answers from the .xlsx files in the "03_Trigger Logs" subfolders (KPI_Format_Converters are excluded).
    
    Input:
        - root_folder: (str)
            Path to the root folder

    Output:
        - labeled_data: (list of 3-element tuples)
            List of 3-element tuples with cluster, note and combined answer values
    """
    root_path = Path(root_folder)
    sheet_name = "Triggerliste"
    note_column = "Note"
    cluster_column = "Cluster"
    combined_answers_column = "Combined answers"

    # List of .xlsx filenames (without path) to exclude
    excluded_files = {}

    column_cluster_values = []
    column_note_values = []
    column_comb_answ_values = []

    # Iterate through year folders (2025, 2026, ...)
    for year_folder in root_path.iterdir():
        if not year_folder.is_dir():
            continue
        # Iterate through first-level subfolders inside each year folder
        for subfolder in year_folder.iterdir():
            if not subfolder.is_dir():
                continue

            trigger_logs_path = subfolder / "03_Trigger Logs"

            if trigger_logs_path.exists() and trigger_logs_path.is_dir():

                # Iterate through xlsx files
                for xlsx_file in trigger_logs_path.glob("*.xlsx"):

                    # Skip files whose name contains any of the excluded substrings
                    if any(excluded in xlsx_file.name for excluded in excluded_files):
                        continue

                    # Read the Excel file
                    df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
                    # Remove rows where either Cluster, Note or Combined answer is empty
                    df = df.dropna(subset=[cluster_column, note_column, combined_answers_column])
                    
                    # Append lists with clusters, notes and combined answers
                    temp_cluster_list = df[cluster_column].dropna().astype(str).tolist()
                    temp_event_list = df[note_column].dropna().astype(str).tolist()
                    temp_comb_ans_list = df[combined_answers_column].dropna().astype(str).tolist()
                    column_cluster_values.extend(temp_cluster_list)
                    column_note_values.extend(temp_event_list)
                    column_comb_answ_values.extend(temp_comb_ans_list)

                    # Clear the lists for next .xlsx file
                    temp_cluster_list.clear()
                    temp_event_list.clear()
                    temp_comb_ans_list.clear()

    # Create a list of 3-element tuples
    labeled_data = list(zip(column_cluster_values, column_note_values, column_comb_answ_values))
    # Sort dictionary alphabetically according to the clusters
    labeled_data_sorted = sorted(labeled_data, key=lambda x: (x[0], x[1]))
    # Remove entries with cluster 'ignore'
    labeled_data_sorted = [item for item in labeled_data_sorted if item[0].lower() != 'ignore']
    return labeled_data_sorted

def get_labeled_data_from_cleaned_dataset(cleaned_file_filepath, exclude_filepath):
    """
    Function that iterates through the cleaned dataset file and extracts all the events, clusters and filepaths 
    (file where the note and cluster is from).
    
    Input:
        - cleaned_file_filepath: (str)
            Path to the cleaned excel file
        - exclude_filepaths: (str)
            Filepath string to exclude from the result.
            Rows where the Filepath column matches the string will be skipped.

    Output:
        - labeled_data: (list of 3-element tuples)
            List of 3-element tuples with cluster, note and filepath
    """
    note_column = "Note"
    cluster_column = "Cluster"
    file_column = "Filepath"

    column_cluster_values = []
    column_note_values = []
    column_filepath_values = []

    # Read the Excel file
    df = pd.read_excel(cleaned_file_filepath)
    # Remove rows where either Cluster, Note or Combined answer is empty
    df = df.dropna(subset=[cluster_column, note_column, file_column])

    # Exclude rows whose filepath matches any of the excluded filepaths
    if exclude_filepath:
        df = df[df[file_column].astype(str) != exclude_filepath]
    
    # Append lists with clusters, notes and filepaths
    temp_cluster_list = df[cluster_column].dropna().astype(str).tolist()
    temp_event_list = df[note_column].dropna().astype(str).tolist()
    temp_filepath_list = df[file_column].dropna().astype(str).tolist()
    column_cluster_values.extend(temp_cluster_list)
    column_note_values.extend(temp_event_list)
    column_filepath_values.extend(temp_filepath_list)

    # Clear the lists for next .xlsx file
    temp_cluster_list.clear()
    temp_event_list.clear()
    temp_filepath_list.clear()

    # Create a list of 3-element tuples
    labeled_data = list(zip(column_cluster_values, column_note_values, column_filepath_values))
    # Sort dictionary alphabetically according to the clusters
    labeled_data_sorted = sorted(labeled_data, key=lambda x: (x[0], x[1]))
    # Remove entries with cluster 'ignore'
    labeled_data_sorted = [item for item in labeled_data_sorted if item[0].lower() != 'ignore']

    return labeled_data_sorted

if __name__ == "__main__":
    # -------------- Adjustables ------------------------------
    model_path   = r"filepath_to_model"
    root_folder  = r"filepath_to_rootfolder"
    labeled_data = get_labeled_data(root_folder)
    df = pd.DataFrame(labeled_data, columns=["Cluster", "Note"])
    # Export to Excel
    df.to_excel("excel_export.xlsx", index=False)
    print('done')
