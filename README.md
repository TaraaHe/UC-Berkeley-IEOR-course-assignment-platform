# UC Berkeley IEOR Course Assignment Platform

A comprehensive web application for optimizing course assignments for IEOR students at UC Berkeley. This platform uses linear programming to efficiently assign students to courses based on their preferences while respecting course capacities and scheduling conflicts.

## 🚀 Features

### 📊 Student Preferences Management
- Load student preferences from Google Forms/Sheets
- Upload preference data via CSV files
- Interactive visualizations of student preferences
- Course popularity analysis with utility-based metrics

### 🔀 Conflict Matrix Management
- Upload conflict matrices via CSV
- Manual schedule entry with intuitive interface
- Automatic conflict detection from course schedules
- Visual conflict matrix heatmap

### 📚 Course Capacity Management
- Set course capacities via sliders
- Upload capacity data via CSV
- Real-time capacity tracking

### 🚀 Optimization Engine
- Linear programming-based optimization using PuLP
- Maximizes student utility while respecting constraints
- Handles course conflicts, capacities, and student limits
- Real-time optimization results with statistics

### ✏️ Manual Assignment Editing
- Manual override capabilities for special cases
- Add/remove courses from student assignments
- Iterative re-optimization with manual constraints
- Comprehensive assignment overview and tracking

### 📊 Advanced Visualizations
- Student preference distributions
- Course enrollment analysis
- Assignment result visualizations
- Interactive plots with customizable display options

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/TaraaHe/UC-Berkeley-IEOR-course-assignment-platform.git
cd UC-Berkeley-IEOR-course-assignment-platform
```

2. Install required packages:
```bash
pip install streamlit pandas numpy plotly pulp gspread gspread-dataframe
```

3. Set up Google Sheets API credentials:
   - Place your `service_account.json` file in the `credentials/` directory
   - Ensure the service account has access to your Google Sheets

## 🚀 Usage

### Running the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Workflow
1. **Load Student Preferences**: Use Google Forms or upload CSV
2. **Set Up Conflict Matrix**: Upload matrix or create from schedules
3. **Configure Course Capacities**: Set limits for each course
4. **Run Optimization**: Generate initial assignments
5. **Manual Editing**: Fine-tune assignments as needed
6. **Re-optimize**: Run optimization with manual constraints

## 📁 Project Structure

```
course_assignment_app/
├── app.py                      # Main Streamlit application
├── optimizer.py                # Linear programming optimization engine
├── google_form_reader.py       # Google Sheets integration
├── credentials/                # API credentials (not tracked)
│   └── service_account.json
├── sample_data/                # Sample data files
└── README.md                   # This file
```

## 🔧 Configuration

### Google Sheets Setup
1. Create a Google Cloud Project
2. Enable Google Sheets API
3. Create a service account
4. Download the JSON credentials file
5. Place it in `credentials/service_account.json`

### Course Configuration
The application supports the following IEOR courses:
- 215, 221, 222, 223, 230, 231, 242A, 242B, 253, 256, 290-1, 290-2

## 📊 Data Formats

### Student Preferences CSV
```csv
Student,Rank1,Rank2,Rank3,Rank4,Rank5
A123,215,221,230,242A,253
B456,221,222,231,242B,256
```

### Conflict Matrix CSV
```csv
Course,215,221,222,223,230,231,242A,242B,253,256,290-1,290-2
215,0,1,0,0,1,0,0,0,0,0,0,0
221,1,0,0,0,0,1,0,0,0,0,0,0
...
```

### Course Capacities CSV
```csv
Course,Capacity
215,75
221,80
222,70
...
```

## 🎯 Optimization Algorithm

The platform uses linear programming to solve the course assignment problem:

### Objective Function
Maximize total student utility: `∑(student, course) utility[student, course] × assignment[student, course]`

### Constraints
- **Capacity Constraints**: Each course cannot exceed its capacity
- **Student Limits**: Each student can be assigned to a limited number of courses
- **Conflict Constraints**: Students cannot be assigned to conflicting courses
- **Preference Constraints**: Students can only be assigned to courses they ranked

### Utility Calculation
Utility for a course is calculated as: `2^(6 - rank)`
- Rank 1: 32 utility points
- Rank 2: 16 utility points
- Rank 3: 8 utility points
- Rank 4: 4 utility points
- Rank 5: 2 utility points

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Tara He** - *Initial work* - [TaraaHe](https://github.com/TaraaHe)

## 🙏 Acknowledgments

- UC Berkeley IEOR Department
- Streamlit team for the amazing web framework
- PuLP developers for the optimization library
- Plotly for interactive visualizations

## 📞 Support

For questions or issues, please open an issue on GitHub or contact the development team. 