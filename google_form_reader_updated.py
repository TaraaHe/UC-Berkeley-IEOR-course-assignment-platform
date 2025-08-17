
"""
google_form_reader.py (Streamlit-Cloud friendly)

- Prefers credentials from Streamlit Secrets: st.secrets["gcp_service_account"]
- Falls back to a local JSON key file path for local development
- Opens a Google Form–linked Google Sheet by spreadsheet ID or title
- Allows worksheet selection by gid or title; defaults to first worksheet
- Returns a cleaned pandas.DataFrame

Usage (Streamlit Cloud):
    df = read_form_responses(
        sheet_name_or_id=st.secrets["sheets"]["spreadsheet_id"],
        worksheet_gid=st.secrets["sheets"].get("worksheet_gid"),   # optional
        worksheet_title=st.secrets["sheets"].get("worksheet_title") # optional
    )

Usage (local dev with JSON file):
    df = read_form_responses(
        json_key_path="credentials/service_account.json",
        sheet_name_or_id="1wKac9HPPr1InxQ_4Tehf2JhjRiiH-UnEMuaSubdD0KU",
        worksheet_title="Form Responses 1"
    )
"""
from __future__ import annotations

import re
import typing as t

import pandas as pd
import gspread

try:
    import streamlit as st
except Exception:  # streamlit not required for non-Streamlit contexts
    st = None  # type: ignore

from google.oauth2.service_account import Credentials
from gspread_dataframe import get_as_dataframe

# Least-privilege scopes (read-only)
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

ID_RE = re.compile(r"^[a-zA-Z0-9-_]{40,}$")  # heuristic for spreadsheet IDs


def _authorize(json_key_path: t.Optional[str] = None) -> gspread.client.Client:
    """
    Return an authorized gspread Client.
    Prefers Streamlit Secrets; falls back to a local JSON key path.
    """
    if st is not None and "gcp_service_account" in st.secrets:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=SCOPES
        )
        return gspread.authorize(creds)

    if json_key_path:
        return gspread.service_account(filename=json_key_path, scopes=SCOPES)

    raise RuntimeError(
        "No Google credentials found. Either add gcp_service_account to st.secrets "
        "or pass json_key_path to read_form_responses()."
    )


def _open_sheet(gc: gspread.client.Client, sheet_name_or_id: t.Optional[str]) -> gspread.Spreadsheet:
    """
    Open spreadsheet by ID (preferred) or by human title.
    If sheet_name_or_id is None, try st.secrets['sheets']['spreadsheet_id'].
    """
    value = sheet_name_or_id
    if value is None and st is not None:
        value = st.secrets.get("sheets", {}).get("spreadsheet_id")

    if not value:
        raise ValueError("sheet_name_or_id was not provided and no value found in st.secrets['sheets']['spreadsheet_id'].")

    # Heuristic: if it looks like an ID, open by key; else open by title
    if ID_RE.match(value):
        return gc.open_by_key(value)
    return gc.open(value)


def _pick_worksheet(
    sh: gspread.Spreadsheet,
    worksheet_gid: t.Optional[t.Union[int, str]] = None,
    worksheet_title: t.Optional[str] = None,
) -> gspread.Worksheet:
    """
    Choose a worksheet by gid or title; default to the first.
    """
    if worksheet_gid is None and st is not None:
        worksheet_gid = st.secrets.get("sheets", {}).get("worksheet_gid")
    if worksheet_title is None and st is not None:
        worksheet_title = st.secrets.get("sheets", {}).get("worksheet_title")

    if worksheet_gid not in (None, ""):
        return sh.get_worksheet_by_id(int(worksheet_gid))  # type: ignore[arg-type]
    if worksheet_title:
        return sh.worksheet(worksheet_title)
    return sh.get_worksheet(0)


def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column headers from Google Forms.
    """
    df = df.copy()
    df.columns = (
        pd.Index(df.columns)
        .map(lambda x: "" if x is None else str(x))
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("\n", " ", regex=False)
        .str.replace(r"[()]", "", regex=True)
        .str.replace(" ", "_")
    )
    return df


def read_form_responses(
    json_key_path: t.Optional[str] = None,
    sheet_name_or_id: t.Optional[str] = None,
    worksheet_title: t.Optional[str] = None,
    worksheet_gid: t.Optional[t.Union[int, str]] = None,
    drop_all_empty: bool = True,
) -> pd.DataFrame:
    """
    Read a Google Form-linked Sheet into a DataFrame.
    Returns a cleaned DataFrame with standardized headers.

    Args:
        json_key_path: Local path to service-account JSON (dev fallback).
        sheet_name_or_id: Spreadsheet ID (preferred) or human title.
        worksheet_title: Worksheet title (tab name).
        worksheet_gid: Worksheet gid (numeric id).
        drop_all_empty: Drop rows that are entirely empty.

    Raises:
        RuntimeError: when credentials are missing.
        ValueError: when the sheet reference is missing.
        gspread exceptions for access/sharing issues.
    """
    gc = _authorize(json_key_path=json_key_path)
    sh = _open_sheet(gc, sheet_name_or_id)
    ws = _pick_worksheet(sh, worksheet_gid=worksheet_gid, worksheet_title=worksheet_title)

    df = get_as_dataframe(ws, evaluate_formulas=True)
    if drop_all_empty:
        df = df.dropna(how="all")

    return _clean_headers(df)
