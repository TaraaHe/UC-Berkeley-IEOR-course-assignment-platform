import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from optimizer import run_optimization
from google_form_reader import read_form_responses
import datetime
# ---------------- Helper Functions ----------------
import re

def normalize_preferences_df(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure the DF has canonical Program and Student columns."""
    df = df.copy()

    # 1) Standardize headers (in case they aren't already cleaned)
    df.columns = (
        pd.Index(df.columns)
        .map(lambda x: "" if x is None else str(x))
        .str.strip().str.replace("\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )

    # 2) Locate likely columns
    prog_col = next((c for c in df.columns if "program" in c.lower()), None)
    id_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ["student id", "uc berkeley id", "sid", "ucb id"])),
        None
    )

    # 3) Build Program (normalize labels)
    if prog_col:
        prog_norm = (
            df[prog_col].astype(str).str.strip().str.lower()
            .replace({
                "master of analytics": "Master of Analytics",
                "analytics": "Master of Analytics",
                "m.analytics": "Master of Analytics",
                "meng": "Master of Engineering",
                "master of engineering": "Master of Engineering",
                "ieor meng": "Master of Engineering",
            })
        )
        # Title-case fallback
        df["Program"] = prog_norm.replace("", pd.NA).fillna(df[prog_col].astype(str).str.strip().str.title())
    else:
        df["Program"] = pd.NA

    # 4) Build Student (prefix by program + ID if present; else index)
    def prefix_for(p):
        s = str(p).lower()
        if "analytic" in s: return "A"
        if "engineer" in s: return "B"
        return "M"  # generic Masters / unknown

    if id_col:
        sid = df[id_col].astype(str).str.strip()
    else:
        sid = df.index.astype(str)

    df["Student"] = [f"{prefix_for(p)}{sid_i}" for p, sid_i in zip(df["Program"], sid)]

    # 5) Also provide a clean ProgramPrefix if you still want it
    df["ProgramPrefix"] = df["Program"].astype(str).str.lower().map(
        lambda s: "A" if "analytic" in s else ("B" if "engineer" in s else "M")
    )

    return df

# ---------------- Sidebar Page Persistence ----------------

# Sidebar navigation
st.sidebar.title("🎯 Course Assignment Platform")
page = st.sidebar.selectbox(
    "Choose a step:",
    ["📊 Student Preferences", "🔀 Conflict Matrix", "📚 Course Capacities", "🚀 Run Optimization", "✏️ Manual Editing"]
)

# Google Sheets Configuration
# Check if Streamlit secrets are configured
has_secrets = hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets and 'sheets' in st.secrets

# Load form config (legacy fallback)
json_path = "credentials/service_account.json"
sheet_name = "IEOR_Preferences"

# Initialize or retrieve data from Streamlit session_state
if 'preferences_df' not in st.session_state:
    st.session_state['preferences_df'] = None
if 'conflict_df' not in st.session_state:
    st.session_state['conflict_df'] = None
if 'capacity_dict' not in st.session_state:
    st.session_state['capacity_dict'] = {}

preferences_df = st.session_state['preferences_df']
conflict_df = st.session_state['conflict_df']
capacity_dict = st.session_state['capacity_dict']

# Define course names globally
course_names = ['215', '221', '222', '223', '230', '231', '242A', '242B', '253', '256', '290-1', '290-2']

def create_preference_visualizations(df, show_debug=False):
    """Create visualizations for student preferences"""
    
    # Debug: Show data structure
    if show_debug:
        st.write("**Debug Info:**")
        st.write(f"Total students: {len(df)}")
        st.write(f"Columns: {list(df.columns)}")
    
    # 1. Distribution of students by program
    program_counts = df['Student'].str[0].value_counts()
    program_labels = {'A': 'Analytics', 'B': 'Engineering', 'M': 'Masters'}
    program_counts.index = [program_labels.get(x, x) for x in program_counts.index]
    
    fig_program = px.pie(
        values=program_counts.values, 
        names=program_counts.index,
        title="Student Distribution by Program"
    )
    
    # Calculate average utility for each course (replacing average rank)
    course_avg_utility = []
    all_courses = set()
    
    # First, collect all unique courses
    for rank in range(1, 6):
        rank_col = f'Rank{rank}'
        if rank_col in df.columns:
            courses_in_rank = df[rank_col].dropna().unique()
            all_courses.update(courses_in_rank)
    
    # Calculate average utility for each course
    for course in all_courses:
        course_utilities = []
        total_students_for_course = 0
        
        for rank in range(1, 6):
            rank_col = f'Rank{rank}'
            if rank_col in df.columns:
                students_at_rank = len(df[df[rank_col] == course])
                if students_at_rank > 0:
                    utility = 2 ** (6 - rank)  # Utility formula from optimizer
                    course_utilities.extend([utility] * students_at_rank)
                    total_students_for_course += students_at_rank
        
        if course_utilities:  # Only add if course was ranked by at least one student
            avg_utility = sum(course_utilities) / len(course_utilities)
            course_avg_utility.append({
                'Course': course,
                'Average_Utility': avg_utility,
                'Total_Students': total_students_for_course
            })
    
    # Predefined course order (used for consistent y-axis)
    course_order = ['215', '221', '222', '223', '230', '231', '242A', '242B', '253', '256', '290-1', '290-2']
    
    avg_utility_df = pd.DataFrame(course_avg_utility)
    if not avg_utility_df.empty:
        # Ensure all courses appear (even if not ranked by anyone)
        missing_courses = set(course_order) - set(avg_utility_df['Course'])
        for mc in missing_courses:
            avg_utility_df = pd.concat([avg_utility_df, pd.DataFrame([{'Course': mc, 'Average_Utility': 0, 'Total_Students': 0}])], ignore_index=True)
        # Preserve the specified order for plotting
        avg_utility_df['Course'] = pd.Categorical(avg_utility_df['Course'], categories=course_order, ordered=True)
        avg_utility_df = avg_utility_df.sort_values('Course')
    
    # Create vertical bar chart for course popularity (higher utility = more popular)
    fig_popularity = px.bar(
        avg_utility_df,
        x='Course',
        y='Average_Utility',
        title="Course Popularity (Average Utility; higher = more popular)",
        text='Average_Utility',
        color_discrete_sequence=['#1f77b4']
    )
    
    # Set categorical x-axis and clean formatting
    fig_popularity.update_layout(xaxis_title='Course', yaxis_title='Average Utility')
    fig_popularity.update_xaxes(type='category')
    fig_popularity.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    
    return fig_program, None, fig_popularity, avg_utility_df

# ================= Assignment Result Visualizations =================
def create_assignment_visualizations(result_df, preferences_df, capacity_dict):
    """Generate visualizations based on the optimization results."""

    # Helper to get prefix
    result_df['Prefix'] = result_df['Student'].str[0]

    # 1. Distribution of number of electives per student
    result_df['Num_Electives'] = result_df['AssignedCourses'].apply(lambda x: 0 if pd.isna(x) or x=='' else len(x.split(',')))

    fig_electives = px.histogram(
        result_df,
        x='Num_Electives',
        color='Prefix',
        barmode='group',
        title='Distribution of Number of Electives per Student',
        category_orders={'Num_Electives': [0,1,2,3,4]}
    )
    fig_electives.update_xaxes(title='Number of Electives Assigned')
    fig_electives.update_yaxes(title='Number of Students')

    # 2. Course Enrollment by Student Prefix with Unused Capacity
    enrollment_records = []
    for _, row in result_df.iterrows():
        student = row['Student']
        prefix = row['Prefix']
        courses = [] if pd.isna(row['AssignedCourses']) or row['AssignedCourses']=='' else [c.strip() for c in row['AssignedCourses'].split(',')]
        for course in courses:
            enrollment_records.append({'Course': course, 'Category': 'Analytics' if prefix=='A' else 'Engineering', 'Count': 1})

    enroll_df = pd.DataFrame(enrollment_records)
    course_order = list(capacity_dict.keys())

    # Aggregate counts
    enroll_agg = enroll_df.groupby(['Course','Category']).sum().reset_index()

    # Add unused capacity rows
    for course, cap in capacity_dict.items():
        total_enrolled = enroll_agg[enroll_agg['Course']==course]['Count'].sum()
        unused = cap - total_enrolled
        if unused < 0:
            unused = 0
        enroll_agg = pd.concat([enroll_agg, pd.DataFrame([{'Course': course, 'Category': 'Unused Capacity', 'Count': unused}])], ignore_index=True)

    fig_course_enroll = px.bar(
        enroll_agg,
        x='Course',
        y='Count',
        color='Category',
        title='Course Enrollment by Student Prefix with Unused Capacity',
        category_orders={'Course': course_order}
    )
    fig_course_enroll.update_xaxes(type='category')

    # 3 & 4. Utility calculations
    # Build utility dictionary
    util_map = {}
    for _, row in preferences_df.iterrows():
        student = row['Student']
        for r in range(1,6):
            course = row.get(f'Rank{r}')
            if pd.notna(course):
                util_map[(student, course)] = 2**(6 - r)

    # Compute utility per student
    utilities = []
    for _, row in result_df.iterrows():
        student = row['Student']
        prefix = row['Prefix']
        courses = [] if pd.isna(row['AssignedCourses']) or row['AssignedCourses']=='' else [c.strip() for c in row['AssignedCourses'].split(',')]
        total_utility = sum([util_map.get((student, c), 0) for c in courses])
        avg_util = total_utility / len(courses) if courses else 0
        utilities.append({'Student': student, 'Prefix': prefix, 'TotalUtility': total_utility, 'AvgUtility': avg_util})

    util_df = pd.DataFrame(utilities)

    # Bar chart total utility by student (grouped by prefix)
    fig_total_util = px.histogram(
        util_df,
        x='TotalUtility',
        color='Prefix',
        nbins=20,
        title='Distribution of Total Utility per Student'
    )
    fig_total_util.update_xaxes(title='Total Utility')
    fig_total_util.update_yaxes(title='Number of Students')

    # Histogram avg utility per student
    fig_avg_util = px.histogram(
        util_df,
        x='AvgUtility',
        color='Prefix',
        nbins=20,
        title='Distribution of Average Utility per Student'
    )
    fig_avg_util.update_xaxes(title='Average Utility per Course')
    fig_avg_util.update_yaxes(title='Number of Students')

    return fig_electives, fig_course_enroll, fig_total_util, fig_avg_util, util_df

def create_enhanced_result_df(result_df, preferences_df):
    """Create enhanced result dataframe with UC Berkeley Student ID and manual editing columns"""
    enhanced_df = result_df.copy()
    
    # Add UC Berkeley Student ID from preferences
    id_mapping = {}
    for _, row in preferences_df.iterrows():
        student = row['Student']
        # Extract UC Berkeley ID (remove prefix A/B/M)
        uc_id = student[1:] if len(student) > 1 else student
        id_mapping[student] = uc_id
    
    enhanced_df['UC_Berkeley_ID'] = enhanced_df['Student'].map(id_mapping)
    
    # Add manual editing columns
    enhanced_df['Enroll'] = enhanced_df['AssignedCourses']  # Copy assigned courses to enroll
    enhanced_df['Drop'] = ''  # Empty drop column
    enhanced_df['Manual_Override'] = False  # Track manually edited rows
    
    # Reorder columns
    enhanced_df = enhanced_df[['Student', 'UC_Berkeley_ID', 'Enroll', 'Drop', 'Manual_Override', 'AssignedCourses']]
    
    return enhanced_df

def run_iterative_optimization(preferences_df, conflict_df, capacity_dict, manual_assignments_df):
    """Run optimization while respecting manual assignments"""
    
    # Get manually assigned students (those with Manual_Override = True)
    manual_students = manual_assignments_df[manual_assignments_df['Manual_Override'] == True]['Student'].tolist()
    
    # Filter preferences to exclude manually assigned students
    filtered_preferences = preferences_df[~preferences_df['Student'].isin(manual_students)].copy()
    
    if len(filtered_preferences) == 0:
        # All students are manually assigned
        return manual_assignments_df
    
    # Calculate remaining capacity after manual assignments
    remaining_capacity = capacity_dict.copy()
    
    for _, row in manual_assignments_df.iterrows():
        if row['Manual_Override']:
            # Calculate final assignment: Original + Enrolled - Dropped
            original_courses = row['AssignedCourses'].split(', ') if pd.notna(row['AssignedCourses']) and row['AssignedCourses'].strip() else []
            original_courses = [c.strip() for c in original_courses if c.strip()]
            
            # Add enrolled courses
            final_courses = original_courses.copy()
            if pd.notna(row['Enroll']) and row['Enroll'].strip():
                enrolled_courses = [c.strip() for c in row['Enroll'].split(',') if c.strip()]
                final_courses.extend(enrolled_courses)
            
            # Remove dropped courses
            if pd.notna(row['Drop']) and row['Drop'].strip():
                dropped_courses = [c.strip() for c in row['Drop'].split(',') if c.strip()]
                final_courses = [c for c in final_courses if c not in dropped_courses]
            
            # Remove duplicates
            final_courses = list(set(final_courses))
            
            # Update capacity for final courses
            for course in final_courses:
                if course in remaining_capacity:
                    remaining_capacity[course] = max(0, remaining_capacity[course] - 1)
    
    # Run optimization on remaining students
    optimized_result = run_optimization(filtered_preferences, conflict_df, remaining_capacity)
    
    # Merge results: keep manual assignments and add optimized assignments
    final_result = []
    
    for _, row in manual_assignments_df.iterrows():
        if row['Manual_Override']:
            # Calculate final assignment: Original + Enrolled - Dropped
            original_courses = row['AssignedCourses'].split(', ') if pd.notna(row['AssignedCourses']) and row['AssignedCourses'].strip() else []
            original_courses = [c.strip() for c in original_courses if c.strip()]
            
            # Add enrolled courses
            final_courses = original_courses.copy()
            if pd.notna(row['Enroll']) and row['Enroll'].strip():
                enrolled_courses = [c.strip() for c in row['Enroll'].split(',') if c.strip()]
                final_courses.extend(enrolled_courses)
            
            # Remove dropped courses
            if pd.notna(row['Drop']) and row['Drop'].strip():
                dropped_courses = [c.strip() for c in row['Drop'].split(',') if c.strip()]
                final_courses = [c for c in final_courses if c not in dropped_courses]
            
            # Remove duplicates
            final_courses = list(set(final_courses))
            
            # Keep calculated final assignment
            final_result.append({
                'Student': row['Student'],
                'AssignedCourses': ', '.join(final_courses) if final_courses else ''
            })
        else:
            # Use optimized assignment if available
            opt_row = optimized_result[optimized_result['Student'] == row['Student']]
            if not opt_row.empty:
                final_result.append({
                    'Student': row['Student'],
                    'AssignedCourses': opt_row.iloc[0]['AssignedCourses']
                })
            else:
                # Keep original if not in optimization
                final_result.append({
                    'Student': row['Student'],
                    'AssignedCourses': row['AssignedCourses']
                })
    
    return pd.DataFrame(final_result)

# ================= Schedule to Conflict Matrix Helper =================
def build_conflict_matrix_from_schedule(schedule_df, course_list=None):
    """Generate a conflict matrix (1 = conflict, 0 = no conflict) from a schedule DataFrame.

    schedule_df columns expected: Course, Days (e.g., 'MWF', 'TTh', 'F'), Start (HH:MM), End (HH:MM)
    """

    def parse_days(days_str):
        days = []
        i = 0
        while i < len(days_str):
            # Look for 'Th'
            if days_str[i:i+2] == 'Th':
                days.append('Th')
                i += 2
            else:
                days.append(days_str[i])
                i += 1
        return set(days)

    def time_to_minutes(t):
        h, m = t.split(':')
        return int(h)*60 + int(m)

    records = {}
    for _, row in schedule_df.iterrows():
        c = row['Course']
        days = parse_days(str(row['Days']).strip())
        start = time_to_minutes(str(row['Start']).strip())
        end = time_to_minutes(str(row['End']).strip())
        records[c] = {'days': days, 'start': start, 'end': end}

    courses = course_list if course_list else list(records.keys())
    matrix = pd.DataFrame(0, index=courses, columns=courses, dtype=int)
    for i, c1 in enumerate(courses):
        for c2 in courses[i+1:]:
            conflict = 0
            if c1 in records and c2 in records:
                d1 = records[c1]['days']
                d2 = records[c2]['days']
                if d1.intersection(d2):
                    s1, e1 = records[c1]['start'], records[c1]['end']
                    s2, e2 = records[c2]['start'], records[c2]['end']
                    if (s1 < e2) and (s2 < e1):
                        conflict = 1
            matrix.loc[c1, c2] = matrix.loc[c2, c1] = conflict
    return matrix

# Sidebar navigation
# st.sidebar.title("🎯 Course Assignment Platform")
# page = st.sidebar.selectbox(
#     "Choose a step:",
#     ["📊 Student Preferences", "🔀 Conflict Matrix", "📚 Course Capacities", "🚀 Run Optimization"]
# )

# Status indicators
st.sidebar.markdown("### 📋 Data Status")
pref_status = "✅ Loaded" if preferences_df is not None else "❌ Not loaded"
conflict_status = "✅ Loaded" if conflict_df is not None else "❌ Not loaded"
capacity_status = "✅ Set" if capacity_dict else "❌ Not set"
optimization_status = "✅ Completed" if 'enhanced_result_df' in st.session_state else "❌ Not run"

st.sidebar.markdown(f"- Preferences: {pref_status}")
st.sidebar.markdown(f"- Conflict Matrix: {conflict_status}")
st.sidebar.markdown(f"- Capacities: {capacity_status}")
st.sidebar.markdown(f"- Optimization: {optimization_status}")

# Page 1: Student Preferences
if page == "📊 Student Preferences":
    st.title("📊 Student Preferences")
    
    tab1, tab2 = st.tabs(["📄 Google Form", "📤 Upload CSV"])
    
    with tab1:
        # Check if secrets are configured
        if not has_secrets:
            st.warning("⚠️ Google Sheets not configured")
            st.info("""
            To use Google Forms integration, you need to configure Streamlit secrets:
            
            1. Create a `.streamlit/secrets.toml` file
            2. Add your Google Cloud service account credentials
            3. Add your Google Sheets configuration
            
            For now, please use the CSV upload option below.
            """)
            
            # Legacy option for users with JSON file
            if st.checkbox("I have a service account JSON file"):
                json_file = st.file_uploader("Upload service account JSON file", type="json")
                sheet_id_input = st.text_input("Enter Google Sheet ID or name", placeholder="e.g., 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms")
                
                if json_file and sheet_id_input:
                    if st.button("Load with JSON file", type="secondary"):
                        with st.spinner("Loading from Google Sheets…"):
                            try:
                                # Save uploaded JSON temporarily
                                import tempfile
                                import os
                                
                                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                                    tmp_file.write(json_file.getvalue().decode('utf-8'))
                                    temp_json_path = tmp_file.name
                                
                                preferences_df = read_form_responses(
                                    json_key_path=temp_json_path,
                                    sheet_name=sheet_id_input
                                )
                                
                                # Clean up temp file
                                os.unlink(temp_json_path)
                                
                                if preferences_df is not None:
                                    # Normalize the preferences dataframe
                                    preferences_df = normalize_preferences_df(preferences_df)
                                    st.session_state["preferences_df"] = preferences_df
                                    st.success(f"Loaded {len(preferences_df):,} responses ✅")
                                    st.dataframe(preferences_df.head(25), use_container_width=True)
                                else:
                                    st.error("Failed to load data from Google Sheets")
                            except Exception as e:
                                st.error(f"Failed to load: {e}")
                                st.info("Please check your JSON file and sheet ID")
        else:
            if st.button("Load Preferences from Google Form", type="primary"):
                with st.spinner("Loading from Google Sheets…"):
                    try:
                        preferences_df = read_form_responses(
                            sheet_name_or_id=st.secrets["sheets"]["spreadsheet_id"],
                            worksheet_gid=st.secrets["sheets"].get("worksheet_gid"),
                            worksheet_title=st.secrets["sheets"].get("worksheet_title")
                        )
                        
                        if preferences_df is not None:
                            # Normalize the preferences dataframe
                            preferences_df = normalize_preferences_df(preferences_df)
                            st.session_state["preferences_df"] = preferences_df
                            st.success(f"Loaded {len(preferences_df):,} responses ✅")
                            st.dataframe(preferences_df.head(25), use_container_width=True)
                        else:
                            st.error("Failed to load data from Google Sheets")
                    except Exception as e:
                        st.error(f"Failed to load: {e}")
                        st.info("""
                        Please check:
                        1. Google Cloud service account is properly configured
                        2. Google Sheet is shared with the service account email
                        3. Correct spreadsheet ID and worksheet configuration
                        """)

    with tab2:
        uploaded_file = st.file_uploader("Upload CSV file with preferences", type="csv")
        if uploaded_file:
            preferences_df = pd.read_csv(uploaded_file)
            st.session_state['preferences_df'] = preferences_df
            st.success("CSV uploaded successfully!")
    
    # Display current preferences
    if preferences_df is not None:
        st.markdown("### 📋 Current Student Preferences")
        st.dataframe(preferences_df.head())
        
        # Visualization controls
        st.markdown("### 📊 Preference Visualizations")
        
        # Toggle for showing visualizations
        show_plots = st.checkbox("Show preference visualizations", value=True)
        
        if show_plots:
            # Individual plot toggles
            col1, col2, col3 = st.columns(3)
            with col1:
                show_program_dist = st.checkbox("Program Distribution", value=True)
            with col2:
                show_popularity_plot = st.checkbox("Course Popularity Plot", value=True)
            with col3:
                show_debug = st.checkbox("Show debug information", value=False)
            
            # Create visualizations
            fig_program, fig_rank, fig_popularity, avg_utility_df = create_preference_visualizations(preferences_df, show_debug)
            
            # Display selected plots
            if show_program_dist:
                st.plotly_chart(fig_program, use_container_width=True)
            
            if show_popularity_plot:
                st.plotly_chart(fig_popularity, use_container_width=True)
            
            # Always show course popularity summary
            st.markdown("### 📈 Course Popularity Summary")
            st.dataframe(avg_utility_df.round(1))
    else:
        st.info("Please load student preferences to continue.")

# Page 2: Conflict Matrix
elif page == "🔀 Conflict Matrix":
    st.title("🔀 Conflict Matrix")
    
    # Two tabs: Upload and Manual Entry
    tab1, tab2 = st.tabs(["📤 Upload Matrix CSV", "⌨️ Enter Schedule Manually"])

    with tab1:
        conflict_file = st.file_uploader("Upload Conflict Matrix CSV", type="csv")
        if conflict_file:
            conflict_df = pd.read_csv(conflict_file, index_col=0)
            st.session_state['conflict_df'] = conflict_df
            st.success("Conflict matrix uploaded successfully!")

    with tab2:
        # Initialize session state for schedules if not exists
        if 'manual_schedules' not in st.session_state:
            st.session_state['manual_schedules'] = {course: {'Days': 'MWF', 'Start': '09:00', 'End': '12:00'} for course in course_names}
        
        # Day options with better labels
        day_options = {
            'MW': '🗓️ Monday & Wednesday',
            'MWF': '🗓️ Monday, Wednesday & Friday', 
            'TTh': '🗓️ Tuesday & Thursday',
            'W': '🗓️ Wednesday Only',
            'F': '🗓️ Friday Only'
        }
        
        st.markdown("#### 📚 Individual Course Schedules")
        
        # Group courses in a more organized way
        cols = st.columns(4)  # 4 columns for better layout
        
        for idx, course in enumerate(course_names):
            col = cols[idx % 4]
            with col:
                # Course header with better styling
                st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><h4 style='margin: 0; color: #1f77b4;'>📖 {course}</h4></div>", unsafe_allow_html=True)
                
                # Days selection with better labels
                current_days = st.session_state['manual_schedules'][course]['Days']
                day_sel = st.selectbox(
                    "Meeting Days", 
                    options=list(day_options.keys()),
                    format_func=lambda x: day_options[x],
                    index=list(day_options.keys()).index(current_days) if current_days in day_options else 0,
                    key=f"day_{course}"
                )
                
                # Time inputs with better formatting
                current_start = datetime.time.fromisoformat(st.session_state['manual_schedules'][course]['Start'])
                current_end = datetime.time.fromisoformat(st.session_state['manual_schedules'][course]['End'])
                
                start_time = st.time_input(
                    "🕐 Start Time", 
                    value=current_start, 
                    step=1800, 
                    key=f"start_{course}",
                    help="Select course start time"
                )
                end_time = st.time_input(
                    "🕐 End Time", 
                    value=current_end, 
                    step=1800, 
                    key=f"end_{course}",
                    help="Select course end time"
                )
                
                # Update session state
                st.session_state['manual_schedules'][course] = {
                    'Days': day_sel,
                    'Start': start_time.strftime('%H:%M'),
                    'End': end_time.strftime('%H:%M')
                }
                
                # Duration indicator
                duration = (datetime.datetime.combine(datetime.date.today(), end_time) - 
                           datetime.datetime.combine(datetime.date.today(), start_time)).total_seconds() / 3600
                if duration > 0:
                    st.caption(f"⏱️ Duration: {duration:.0f} hour{'s' if duration != 1 else ''}")
                else:
                    st.error("❌ End time must be after start time")
        
        st.markdown("---")
        
        # Generate button with better styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Generate Conflict Matrix", type="primary", use_container_width=True):
                # Build schedule dataframe from session state
                schedule_data = []
                for course, schedule in st.session_state['manual_schedules'].items():
                    schedule_data.append({
                        'Course': course,
                        'Days': schedule['Days'],
                        'Start': schedule['Start'],
                        'End': schedule['End']
                    })
                
                schedule_df = pd.DataFrame(schedule_data)
                
                try:
                    with st.spinner("🔄 Analyzing course schedules and detecting conflicts..."):
                        conflict_df = build_conflict_matrix_from_schedule(schedule_df)
                        st.session_state['conflict_df'] = conflict_df
                    
                    st.success("✅ Conflict matrix generated successfully!")
                    
                    # Show a preview of conflicts detected
                    conflicts = []
                    for i in range(len(conflict_df)):
                        for j in range(i+1, len(conflict_df)):
                            if conflict_df.iloc[i, j] == 1:
                                conflicts.append(f"{conflict_df.index[i]} ↔️ {conflict_df.columns[j]}")
                    
                    if conflicts:
                        st.warning(f"⚠️ {len(conflicts)} conflict(s) detected: {', '.join(conflicts[:5])}")
                        if len(conflicts) > 5:
                            st.caption(f"... and {len(conflicts) - 5} more conflicts")
                    else:
                        st.info("✨ No scheduling conflicts detected!")
                        
                except Exception as e:
                    st.error(f"❌ Failed to generate conflict matrix: {e}")
        
        # Show current schedule summary in an expandable section
        with st.expander("📋 Current Schedule Summary", expanded=False):
            summary_df = pd.DataFrame([
                {
                    'Course': course,
                    'Days': day_options.get(schedule['Days'], schedule['Days']),
                    'Time': f"{schedule['Start']} - {schedule['End']}"
                }
                for course, schedule in st.session_state['manual_schedules'].items()
            ])
            st.dataframe(summary_df, use_container_width=True)

    # Display current conflict matrix
    if conflict_df is not None:
        st.markdown("### 📋 Current Conflict Matrix")
        st.dataframe(conflict_df)
        
        # Heatmap
        st.markdown("### 🔥 Conflict Matrix Heatmap")
        fig_heatmap = px.imshow(
            conflict_df.astype(float),
            title="Course Conflicts (1 = Conflict, 0 = No Conflict)",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Download button
        csv_conf = conflict_df.to_csv().encode("utf-8")
        st.download_button("📥 Download Conflict Matrix CSV", csv_conf, "conflict_matrix.csv", "text/csv")
    else:
        st.info("Please upload a matrix or generate one from schedule to continue.")

# Page 3: Course Capacities
elif page == "📚 Course Capacities":
    st.title("📚 Course Capacities")
    
    cap_upload = st.file_uploader("Upload Course Capacities CSV", type="csv")
    if cap_upload:
        cap_df = pd.read_csv(cap_upload)
        capacity_dict = dict(zip(cap_df['Course'], cap_df['Capacity']))
        st.session_state['capacity_dict'] = capacity_dict
        st.success("Course capacities uploaded successfully!")
    
    # Manual capacity setting
    st.markdown("### 📝 Set Capacities Manually")
    with st.form("Set Capacities Manually"):
        capacity_inputs = {}
        cols = st.columns(2)
        for idx, course in enumerate(course_names):
            with cols[idx % 2]:
                capacity_inputs[course] = st.slider(
                    f"{course} Capacity",
                    min_value=0,
                    max_value=200,
                    value=capacity_dict.get(course, 75) if isinstance(capacity_dict, dict) else 75,
                    step=1,
                    key=f"cap_{course}"
                )
        
        if st.form_submit_button("Save Capacities"):
            capacity_dict = capacity_inputs
            st.session_state['capacity_dict'] = capacity_dict
            st.success("Capacities saved successfully!")
    

# Page 4: Run Optimization
elif page == "🚀 Run Optimization":
    st.title("🚀 Run Optimization")
    
    # Check if all data is loaded
    if preferences_df is not None and conflict_df is not None and capacity_dict:
        st.success("✅ All data loaded! Ready to optimize.")
        
        # Show summary
        st.markdown("### 📊 Optimization Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Students", len(preferences_df))
        with col2:
            st.metric("Courses", len(conflict_df.columns))
        with col3:
            st.metric("Total Capacity", sum(capacity_dict.values()))
        
        if st.button("🚀 Optimize Course Assignment", type="primary"):
            try:
                with st.spinner("Running optimization..."):
                    result_df = run_optimization(preferences_df, conflict_df, capacity_dict)
                
                st.success("✅ Optimization completed!")
                st.dataframe(result_df)
                
                # Store original result for reset functionality
                st.session_state['original_result_df'] = result_df.copy()
                
                # Initialize enhanced result dataframe for manual editing
                st.session_state['enhanced_result_df'] = create_enhanced_result_df(result_df, preferences_df)
                
                # Store the very first optimization result as original assignment
                st.session_state['original_assignments'] = result_df.set_index('Student')['AssignedCourses'].to_dict()
                
                # Download results
                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Assignment CSV", 
                    csv, 
                    "assignment_results.csv", 
                    "text/csv"
                )
                
                # Show assignment statistics
                st.markdown("### 📈 Assignment Statistics")
                assignments_per_student = result_df['AssignedCourses'].str.count(',').add(1)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Students Assigned", len(result_df))
                with col2:
                    st.metric("Avg Courses/Student", round(assignments_per_student.mean(), 2))
                with col3:
                    st.metric("Total Assignments", assignments_per_student.sum())
                
                # ================= Assignment Result Visualizations =================
                st.markdown("### 📊 Assignment Result Visualizations")
                fig_electives, fig_course_enroll, fig_total_util, fig_avg_util, util_df = create_assignment_visualizations(result_df, preferences_df, capacity_dict)

                st.plotly_chart(fig_electives, use_container_width=True)
                st.plotly_chart(fig_course_enroll, use_container_width=True)
                st.plotly_chart(fig_avg_util, use_container_width=True)
                
                # Display utility table (optional)
                with st.expander("Utility per Student Table"):
                    st.dataframe(util_df)

            except Exception as e:
                st.error(f"❌ Optimization failed: {e}")
    else:
        st.error("❌ Please complete all previous steps before running optimization.")
        st.markdown("**Missing:**")
        if preferences_df is None:
            st.markdown("- ❌ Student Preferences")
        if conflict_df is None:
            st.markdown("- ❌ Conflict Matrix")
        if not capacity_dict:
            st.markdown("- ❌ Course Capacities")

# Page 5: Manual Editing
elif page == "✏️ Manual Editing":
    st.title("✏️ Manual Editing")
    
    # Check if optimization has been run
    if 'enhanced_result_df' not in st.session_state:
        st.error("❌ Please run optimization first before manual editing.")
        st.info("Go to '🚀 Run Optimization' page to generate initial assignments.")
    else:
        st.success("✅ Optimization results available for manual editing.")
        
        # Get the enhanced result dataframe
        enhanced_df = st.session_state['enhanced_result_df']
        
        # User Guide Section
        with st.expander("📖 User Guide: How to Use Manual Assignment Editor", expanded=False):
            st.markdown("""
            #### 🎯 Purpose
            The Manual Assignment Editor allows you to:
            - Manually adjust course assignments for specific students
            - Add or remove courses from student schedules
            - Handle special cases that the automatic optimizer might miss
            
            #### 📋 Instructions
            - **Enroll Additional Courses**: Enter courses to ADD to the student's assignment
            - **Drop Courses from Original**: Enter courses to REMOVE from the original assignment
            - **Manual Override**: Check to protect this assignment from re-optimization
            
            #### 💡 Examples
            - **Adding courses**: Enter "215, 221" in Enroll field
            - **Removing courses**: Enter "230" in Drop field to remove course 230
            - **Final result**: Original + Enrolled - Dropped = Final Assignment
            """)
        
        # Display current assignments overview
        st.markdown("### 📋 Current Assignments Overview")
        
        # Show summary of current state
        manual_count = len(enhanced_df[enhanced_df['Manual_Override'] == True])
        auto_count = len(enhanced_df[enhanced_df['Manual_Override'] == False])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Manual Assignments", manual_count)
        with col2:
            st.metric("Auto Assignments", auto_count)
        with col3:
            st.metric("Total Students", len(enhanced_df))
        
        # Display read-only table of current assignments with calculated final assignment
        display_data = []
        original_assignments = st.session_state.get('original_assignments', {})
        
        for _, row in enhanced_df.iterrows():
            # Get the very first original assignment
            original_assignment = original_assignments.get(row['Student'], row['AssignedCourses'])
            original_courses = original_assignment.split(', ') if pd.notna(original_assignment) and original_assignment.strip() else []
            original_courses = [c.strip() for c in original_courses if c.strip()]
            
            # Add enrolled courses
            final_courses = original_courses.copy()
            if pd.notna(row['Enroll']) and row['Enroll'].strip():
                enroll_courses = [c.strip() for c in row['Enroll'].split(',') if c.strip()]
                final_courses.extend(enroll_courses)
            
            # Remove dropped courses
            if pd.notna(row['Drop']) and row['Drop'].strip():
                drop_courses = [c.strip() for c in row['Drop'].split(',') if c.strip()]
                final_courses = [c for c in final_courses if c not in drop_courses]
            
            # Remove duplicates
            final_courses = list(set(final_courses))
            
            display_data.append({
                'Student ID': row['Student'],
                'UC Berkeley ID': row['UC_Berkeley_ID'],
                'Final Assignment': ', '.join(final_courses) if final_courses else 'No courses',
                'Manually Edited': 'Yes' if row['Manual_Override'] else 'No',
                'Original Assignment': original_assignment
            })
        
        display_df = pd.DataFrame(display_data)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Student Editor - Enhanced Approach
        st.markdown("### ✏️ Edit Student Assignments")
        st.markdown("**📋 Instructions:** Enroll = Additional courses to add | Drop = Courses to remove from original assignment")
        
        # Initialize editing state with unique keys per student
        if 'editing_student' not in st.session_state:
            st.session_state['editing_student'] = enhanced_df['Student'].iloc[0] if len(enhanced_df) > 0 else None
        
        # Initialize student-specific editing state
        selected_student = st.session_state['editing_student']
        if f'edit_enroll_{selected_student}' not in st.session_state:
            st.session_state[f'edit_enroll_{selected_student}'] = ""
        if f'edit_drop_{selected_student}' not in st.session_state:
            st.session_state[f'edit_drop_{selected_student}'] = ""
        if f'edit_override_{selected_student}' not in st.session_state:
            st.session_state[f'edit_override_{selected_student}'] = False
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Student selection
            student_options = enhanced_df['Student'].tolist()
            selected_student = st.selectbox(
                "🎓 Select Student to Edit", 
                student_options, 
                key="student_selector_manual",
                index=student_options.index(st.session_state['editing_student']) if st.session_state['editing_student'] in student_options else 0
            )
            
            # Update editing state when student changes
            if selected_student != st.session_state['editing_student']:
                st.session_state['editing_student'] = selected_student
                current_row = enhanced_df[enhanced_df['Student'] == selected_student].iloc[0]
                st.session_state[f'edit_enroll_{selected_student}'] = current_row['Enroll'] if pd.notna(current_row['Enroll']) else ""
                st.session_state[f'edit_drop_{selected_student}'] = current_row['Drop'] if pd.notna(current_row['Drop']) else ""
                st.session_state[f'edit_override_{selected_student}'] = current_row['Manual_Override'] if pd.notna(current_row['Manual_Override']) else False
                st.rerun()
            
            # Get current data for selected student
            current_row = enhanced_df[enhanced_df['Student'] == selected_student].iloc[0]
            
            # Show current info
            st.markdown(f"**UC Berkeley ID:** {current_row['UC_Berkeley_ID']}")
            original_assignments = st.session_state.get('original_assignments', {})
            original_assignment = original_assignments.get(selected_student, current_row['AssignedCourses'])
            st.markdown(f"**Original Assignment:** {original_assignment}")
            
            # Enroll courses input (Additional courses to add)
            new_enroll = st.text_area(
                "📚 Enroll Additional Courses", 
                value=st.session_state[f'edit_enroll_{selected_student}'],
                help="Enter ADDITIONAL courses to add (separated by commas, e.g., '215, 221')",
                height=80,
                key=f"enroll_input_manual_{selected_student}"
            )
        
        with col2:
            # Drop courses input (Remove from original assignment)
            new_drop = st.text_area(
                "🗑️ Drop Courses from Original", 
                value=st.session_state[f'edit_drop_{selected_student}'],
                help="Enter courses to REMOVE from original assignment (separated by commas, e.g., '230, 242B')",
                height=80,
                key=f"drop_input_manual_{selected_student}"
            )
            
            # Manual override checkbox
            new_override = st.checkbox(
                "🔒 Manual Override", 
                value=st.session_state[f'edit_override_{selected_student}'],
                help="Check to protect this assignment from re-optimization",
                key=f"override_input_manual_{selected_student}"
            )
            
            # Show current final assignment preview
            st.markdown("**📊 Final Assignment Preview:**")
            original_assignments = st.session_state.get('original_assignments', {})
            original_assignment = original_assignments.get(selected_student, current_row['AssignedCourses'])
            original_courses = original_assignment.split(', ') if pd.notna(original_assignment) and original_assignment.strip() else []
            original_courses = [c.strip() for c in original_courses if c.strip()]
            
            # Calculate final assignment
            final_courses = original_courses.copy()
            
            # Add enrolled courses
            if new_enroll.strip():
                enroll_courses = [c.strip() for c in new_enroll.split(',') if c.strip()]
                final_courses.extend(enroll_courses)
            
            # Remove dropped courses
            if new_drop.strip():
                drop_courses = [c.strip() for c in new_drop.split(',') if c.strip()]
                final_courses = [c for c in final_courses if c not in drop_courses]
            
            # Remove duplicates
            final_courses = list(set(final_courses))
            
            if final_courses:
                st.success(f"**Final:** {', '.join(final_courses)}")
            else:
                st.warning("**Final:** No courses assigned")
        
        # Action buttons
        col1_btn, col2_btn, col3_btn = st.columns(3)
        with col1_btn:
            if st.button("✅ Update Student", type="primary"):
                # Update the dataframe using current widget values
                mask = st.session_state['enhanced_result_df']['Student'] == selected_student
                st.session_state['enhanced_result_df'].loc[mask, 'Enroll'] = new_enroll
                st.session_state['enhanced_result_df'].loc[mask, 'Drop'] = new_drop
                st.session_state['enhanced_result_df'].loc[mask, 'Manual_Override'] = new_override
                
                # Update session state for consistency
                st.session_state[f'edit_enroll_{selected_student}'] = new_enroll
                st.session_state[f'edit_drop_{selected_student}'] = new_drop
                st.session_state[f'edit_override_{selected_student}'] = new_override
                
                # Track updates
                if 'recent_updates' not in st.session_state:
                    st.session_state['recent_updates'] = []
                st.session_state['recent_updates'].append(f"Updated {selected_student}")
                
                st.success(f"✅ Updated assignment for {selected_student}")
        
        with col2_btn:
            if st.button("🗑️ Clear All Edits"):
                # Clear only the edits, not the original assignment
                st.session_state[f'edit_enroll_{selected_student}'] = ""
                st.session_state[f'edit_drop_{selected_student}'] = ""
                st.session_state[f'edit_override_{selected_student}'] = False
                
                # Update the dataframe
                mask = st.session_state['enhanced_result_df']['Student'] == selected_student
                st.session_state['enhanced_result_df'].loc[mask, 'Enroll'] = ""
                st.session_state['enhanced_result_df'].loc[mask, 'Drop'] = ""
                st.session_state['enhanced_result_df'].loc[mask, 'Manual_Override'] = False
                
                # Track updates
                if 'recent_updates' not in st.session_state:
                    st.session_state['recent_updates'] = []
                st.session_state['recent_updates'].append(f"Cleared edits for {selected_student}")
                
                st.success(f"🗑️ Cleared all edits for {selected_student}")
                st.rerun()
        
        with col3_btn:
            if st.button("🔄 Refresh Overview"):
                st.rerun()
        
        # Show recent updates
        if 'recent_updates' in st.session_state and st.session_state['recent_updates']:
            st.markdown("#### 📝 Recent Updates")
            for update in st.session_state['recent_updates'][-5:]:
                st.caption(f"• {update}")
        
        # Final Actions
        st.markdown("---")
        st.markdown("### 🚀 Final Actions")
        st.markdown("After editing students, choose your next action:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Re-run Optimization", type="primary", use_container_width=True):
                try:
                    with st.spinner("Re-optimizing with manual constraints..."):
                        # Get current enhanced dataframe
                        current_df = st.session_state['enhanced_result_df']
                        
                        # Get current enhanced dataframe (no pre-processing needed)
                        processed_df = current_df.copy()
                        
                        # Run iterative optimization
                        new_result = run_iterative_optimization(preferences_df, conflict_df, capacity_dict, processed_df)
                        
                        # Update enhanced dataframe with new results
                        updated_enhanced = create_enhanced_result_df(new_result, preferences_df)
                        # Preserve manual overrides and their original edit data
                        for idx, row in processed_df.iterrows():
                            if row['Manual_Override']:
                                mask = updated_enhanced['Student'] == row['Student']
                                if mask.any():
                                    # Keep the original edit data (Enroll, Drop) but update AssignedCourses with the new result
                                    updated_enhanced.loc[mask, 'Enroll'] = row['Enroll']
                                    updated_enhanced.loc[mask, 'Drop'] = row['Drop']
                                    updated_enhanced.loc[mask, 'Manual_Override'] = True
                        
                        st.session_state['enhanced_result_df'] = updated_enhanced
                        st.success("✅ Re-optimization completed!")
                        
                        # Show updated results in a table
                        st.markdown("### 📊 Updated Optimization Results")
                        results_data = []
                        original_assignments = st.session_state.get('original_assignments', {})
                        
                        for _, row in updated_enhanced.iterrows():
                            # Get the very first original assignment
                            original_assignment = original_assignments.get(row['Student'], row['AssignedCourses'])
                            original_courses = original_assignment.split(', ') if pd.notna(original_assignment) and original_assignment.strip() else []
                            original_courses = [c.strip() for c in original_courses if c.strip()]
                            
                            # Add enrolled courses
                            final_courses = original_courses.copy()
                            if pd.notna(row['Enroll']) and row['Enroll'].strip():
                                enroll_courses = [c.strip() for c in row['Enroll'].split(',') if c.strip()]
                                final_courses.extend(enroll_courses)
                            
                            # Remove dropped courses
                            if pd.notna(row['Drop']) and row['Drop'].strip():
                                drop_courses = [c.strip() for c in row['Drop'].split(',') if c.strip()]
                                final_courses = [c for c in final_courses if c not in drop_courses]
                            
                            # Remove duplicates
                            final_courses = list(set(final_courses))
                            
                            results_data.append({
                                'Student ID': row['Student'],
                                'UC Berkeley ID': row['UC_Berkeley_ID'],
                                'Final Assignment': ', '.join(final_courses) if final_courses else 'No courses',
                                'Manually Edited': 'Yes' if row['Manual_Override'] else 'No',
                                'Original Assignment': original_assignment
                            })
                        
                        results_display = pd.DataFrame(results_data)
                        st.dataframe(results_display, use_container_width=True, hide_index=True)
                        
                        # Show summary of changes
                        st.markdown("### 📈 Re-optimization Summary")
                        new_manual_count = len(updated_enhanced[updated_enhanced['Manual_Override'] == True])
                        new_auto_count = len(updated_enhanced[updated_enhanced['Manual_Override'] == False])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Manual Assignments", new_manual_count, delta=new_manual_count - manual_count)
                        with col2:
                            st.metric("Auto Assignments", new_auto_count, delta=new_auto_count - auto_count)
                        with col3:
                            st.metric("Total Students", len(updated_enhanced))
                        
                        # Store flag that re-optimization was completed
                        st.session_state['reoptimization_completed'] = True
                        st.session_state['last_optimization_results'] = updated_enhanced
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Re-optimization failed: {e}")
        
        # Show visualizations if re-optimization was completed
        if st.session_state.get('reoptimization_completed', False):
            st.markdown("---")
            st.markdown("### 📊 Assignment Result Visualizations")
            
            # Create visualizations dropdown
            show_viz = st.checkbox("Show assignment visualizations", value=False)
            
            if show_viz:
                # Get the last optimization results
                updated_enhanced = st.session_state.get('last_optimization_results')
                if updated_enhanced is not None:
                    # Create visualizations for the updated results
                    # First, we need to create a result_df format from the updated_enhanced
                    result_df_for_viz = updated_enhanced[['Student', 'AssignedCourses']].copy()
                    # Calculate final assignments for visualization
                    for idx, row in result_df_for_viz.iterrows():
                        original_assignments = st.session_state.get('original_assignments', {})
                        original_assignment = original_assignments.get(row['Student'], row['AssignedCourses'])
                        original_courses = original_assignment.split(', ') if pd.notna(original_assignment) and original_assignment.strip() else []
                        original_courses = [c.strip() for c in original_courses if c.strip()]
                        
                        # Add enrolled courses
                        final_courses = original_courses.copy()
                        if pd.notna(updated_enhanced.loc[idx, 'Enroll']) and updated_enhanced.loc[idx, 'Enroll'].strip():
                            enroll_courses = [c.strip() for c in updated_enhanced.loc[idx, 'Enroll'].split(',') if c.strip()]
                            final_courses.extend(enroll_courses)
                        
                        # Remove dropped courses
                        if pd.notna(updated_enhanced.loc[idx, 'Drop']) and updated_enhanced.loc[idx, 'Drop'].strip():
                            drop_courses = [c.strip() for c in updated_enhanced.loc[idx, 'Drop'].split(',') if c.strip()]
                            final_courses = [c for c in final_courses if c not in drop_courses]
                        
                        # Remove duplicates
                        final_courses = list(set(final_courses))
                        result_df_for_viz.at[idx, 'AssignedCourses'] = ', '.join(final_courses) if final_courses else ''
                    
                    # Create visualizations
                    fig_electives, fig_course_enroll, fig_total_util, fig_avg_util, util_df = create_assignment_visualizations(result_df_for_viz, preferences_df, capacity_dict)
                    
                    # Individual plot toggles
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        show_electives = st.checkbox("Distribution of Electives per Student", value=True)
                    with col2:
                        show_enrollment = st.checkbox("Course Enrollment by Student Prefix", value=True)
                    with col3:
                        show_avg_util = st.checkbox("Distribution of Average Utility per Student", value=True)
                    
                    # Display selected plots
                    if show_electives:
                        st.plotly_chart(fig_electives, use_container_width=True)
                    if show_enrollment:
                        st.plotly_chart(fig_course_enroll, use_container_width=True)
                    if show_avg_util:
                        st.plotly_chart(fig_avg_util, use_container_width=True)
        
        with col2:
            # Download enhanced results
            current_df = st.session_state['enhanced_result_df']
            enhanced_csv = current_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Enhanced Results", 
                enhanced_csv, 
                "enhanced_assignment_results.csv", 
                "text/csv",
                use_container_width=True
            )
        
        with col3:
            if st.button("🔄 Reset to Original", use_container_width=True):
                # Get the original result_df from the optimization
                if 'original_result_df' in st.session_state:
                    st.session_state['enhanced_result_df'] = create_enhanced_result_df(st.session_state['original_result_df'], preferences_df)
                # Clear recent updates
                if 'recent_updates' in st.session_state:
                    st.session_state['recent_updates'] = []
                # Clear re-optimization flag
                if 'reoptimization_completed' in st.session_state:
                    del st.session_state['reoptimization_completed']
                if 'last_optimization_results' in st.session_state:
                    del st.session_state['last_optimization_results']
                st.success("Reset to original optimization results!")
                st.rerun()
        
        # Show capacity usage
        current_df = st.session_state['enhanced_result_df']
        remaining_capacity_total = sum(capacity_dict.values())
        used_capacity = 0
        for _, row in current_df.iterrows():
            if pd.notna(row['Enroll']) and row['Enroll'].strip():
                used_capacity += len([c for c in row['Enroll'].split(',') if c.strip()])
        
        st.markdown("#### 📊 Capacity Usage")
        progress = used_capacity / remaining_capacity_total if remaining_capacity_total > 0 else 0
        st.progress(progress)
        st.caption(f"Used: {used_capacity} / {remaining_capacity_total} total capacity ({progress:.1%})")





