import pandas as pd
from pulp import *

def run_optimization(preferences_df, conflict_df, capacity_dict):
    students = preferences_df['Student'].tolist()
    courses = conflict_df.columns.tolist()

    utility = {}
    for _, row in preferences_df.iterrows():
        s = row['Student']
        for rank in range(1, 6):
            c = row.get(f'Rank{rank}')
            if pd.notna(c):
                utility[(s, c)] = 2 ** (6 - rank)

    conflict_pairs = [(courses[i], courses[j]) for i in range(len(courses)) for j in range(i+1, len(courses)) if conflict_df.loc[courses[i], courses[j]] == 1]

    model = LpProblem("Course_Assignment", LpMaximize)
    x = LpVariable.dicts('x', ((s, c) for s in students for c in courses), cat=LpBinary)

    # Objective
    model += lpSum(utility.get((s, c), 0) * x[(s, c)] for s in students for c in courses)

    # Constraints
    # Capacity constraints
    for c in courses:
        model += lpSum(x[(s, c)] for s in students) <= capacity_dict.get(c, 0)

    # elective count constraints
    for s in students:
        sel = lpSum(x[(s, c)] for c in courses)
        model += sel <= 4
        if s.startswith("A"):
            model += sel >= 3
        elif s.startswith("M"):
            model += sel >= 2

    # no‐conflicts constraints
    for s in students:
        for c1, c2 in conflict_pairs:
            model += x[(s, c1)] + x[(s, c2)] <= 1

    # A-students forbidden from 242A and 215
    for s in students:
        if s.startswith("A"):
            for forbidden in ["242A", "215"]:
                model += x[(s, forbidden)] == 0

    # Fairness - each student must get at least 2 of Top 3 choices
    for _, row in preferences_df.iterrows():
        s = row['Student']
        top_courses = []
        for rank in range(1, 4):
            c = row.get(f'Rank{rank}')
            if pd.notna(c) and c in courses:
                if s.startswith("A") and c in ["215", "242A"]:
                    continue
                top_courses.append(c)
        if len(top_courses) >= 2:
            model += lpSum(x[(s, c)] for c in top_courses) >= 2

    model.solve(PULP_CBC_CMD(msg=0))

    assignments = []
    for s in students:
        assigned_courses = [c for c in courses if value(x[(s, c)]) == 1]
        assignments.append({'Student': s, 'AssignedCourses': ", ".join(assigned_courses)})

    return pd.DataFrame(assignments)
