"""
helper_functions.py

Script with some helper functions for the other .py files.

Author: Alexander Straub
"""

import pandas as pd

def extract_events_from_single_xlsx(xlsx_file):
    """
    Function that iterates through a folder and extracts all the events and combined answers
    from the "Note" and "Combined Answers" column of each .xlsx file. Those events are then stored in a list.
    
    Input: 
        - xlsx_file: (str)
            path to the .xlsx file

    Output:
        - event_list: (list) 
            list of all the events
        - combined_answer_list: (list)
            list of all the combined answer values
    """
    sheet_name = "Triggerliste"
    event_list = []
    combined_answer_list = []

    df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
    note_column = "Note"
    comb_ans_col = "Combined answers"
    temp_event_list = df[note_column].dropna().astype(str).tolist()
    event_list.extend(temp_event_list)
    temp_comb_ans_list = df[comb_ans_col].dropna().astype(str).tolist()
    combined_answer_list.extend(temp_comb_ans_list)
    
    return event_list, combined_answer_list

def extract_events_and_clusters_from_single_xlsx(xlsx_file):
    """
    Function that iterates through a folder and extracts all the events and clusters from 
    the "Note" and "Cluster" column of each .xlsx file. Those events are then stored in a list.
    
    Input: 
        - xlsx_file: (str)
            path to the .xlsx file

    Output:
        - event_list: (list) 
            list of all the events
        - cluster_list: (list) 
            list of all the clusters
    """
    sheet_name = "Triggerliste"
    event_list = []
    cluster_list = []

    df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
    note_column = "Note"
    temp_event_list = df[note_column].dropna().astype(str).tolist()
    event_list.extend(temp_event_list)

    cluster_column = "Cluster"
    temp_cluster_list = df[cluster_column].fillna("").astype(str).tolist()  # keep nan entries
    cluster_list.extend(temp_cluster_list)
    cluster_list = cluster_list[:294]                                       # cut the list to match event length | e.g. 'CLR reached' not needed
                                                                            # old: so i dont really know why i put [:294] here -> wont be needed

    
    return event_list, cluster_list