import gspread
import pandas as pd
from gspread_dataframe import get_as_dataframe
import streamlit as st

def read_form_responses(json_key_path=None, sheet_name=None, sheet_name_or_id=None, worksheet_gid=None, worksheet_title=None):
    """
    Read form responses from Google Sheets.
    
    Args:
        json_key_path: Path to service account JSON file (legacy)
        sheet_name: Name of the sheet (legacy)
        sheet_name_or_id: Sheet ID or name (new)
        worksheet_gid: Worksheet GID (new)
        worksheet_title: Worksheet title (new)
    """
    try:
        # Use Streamlit secrets if available, otherwise use file path
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            # Use Streamlit secrets
            gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
            sheet_id = sheet_name_or_id
        elif json_key_path:
            # Use file path (legacy method)
            gc = gspread.service_account(filename=json_key_path)
            sheet_id = sheet_name
        else:
            raise ValueError("Either Streamlit secrets must be configured or json_key_path must be provided")
        
        # Open the sheet
        if sheet_id:
            sheet = gc.open_by_key(sheet_id) if sheet_id.startswith('1') else gc.open(sheet_id)
        else:
            raise ValueError("Sheet ID or name must be provided")
        
        # Get the worksheet
        if worksheet_gid:
            worksheet = sheet.get_worksheet_by_id(int(worksheet_gid))
        elif worksheet_title:
            worksheet = sheet.worksheet(worksheet_title)
        else:
            worksheet = sheet.get_worksheet(0)
        
        if not worksheet:
            raise ValueError("Could not find worksheet")
        
        # Get data as DataFrame
        df = get_as_dataframe(worksheet, evaluate_formulas=True).dropna(how='all')
        
        if df.empty:
            raise ValueError("No data found in the worksheet")
        
        # Clean up column names
        df.columns = df.columns.str.strip().str.replace("\n", " ").str.replace(" ", "_")
        
        # Define mapping of rank labels to simplified rank keys
        rank_labels = {
            "Rank 1": "Rank1",
            "Rank 2": "Rank2", 
            "Rank 3": "Rank3",
            "Rank 4": "Rank4",
            "Rank 5": "Rank5"
        }
        
        # Identify course ranking columns by program
        analytics_cols = [col for col in df.columns if ".1" not in col and "Course_Preference_Rank" in col]
        engineering_cols = [col for col in df.columns if ".1" in col and "Course_Preference_Rank" in col]
        
        # Extract student ID and program info
        student_info = df[["UC_Berkeley_Student_ID", "What_is_your_program?"]].copy()
        student_info["UC_Berkeley_Student_ID"] = student_info["UC_Berkeley_Student_ID"].astype(str).str.strip()
        student_info["Program"] = student_info["What_is_your_program?"].str.strip()
        
        # Generate new Student ID with prefix
        student_info["Student"] = student_info.apply(
            lambda row: ("A" if row["Program"] == "Master of Analytics" else "B") + row["UC_Berkeley_Student_ID"], axis=1
        )
        
        # Initialize result DataFrame
        result_df = pd.DataFrame(student_info["Student"])
        
        # Process rankings for each student
        for i, row in df.iterrows():
            program = row["What_is_your_program?"].strip()
            rank_dict = {}
            
            # Pick columns based on program
            columns_to_check = analytics_cols if program == "Master of Analytics" else engineering_cols
            
            for col in columns_to_check:
                course_code = col.split("[")[-1].replace("]", "").replace(".1", "")
                rank_val = row[col]
                if pd.notna(rank_val) and rank_val in rank_labels:
                    rank_key = rank_labels[rank_val]
                    rank_dict[rank_key] = course_code
            
            # Assign rank results to output DataFrame
            for rank in ["Rank1", "Rank2", "Rank3", "Rank4", "Rank5"]:
                result_df.loc[i, rank] = rank_dict.get(rank, None)
        
        return result_df
        
    except Exception as e:
        st.error(f"Error reading Google Form: {str(e)}")
        st.info("Please check your Google Sheets configuration and try again.")
        return None
