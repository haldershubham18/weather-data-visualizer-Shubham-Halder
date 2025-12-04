# weather_analyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

class WeatherDataAnalyzer:
    def __init__(self, file_path):
        """
        Initialize the WeatherDataAnalyzer with the path to the CSV file.
        
        Parameters:
        file_path (str): Path to the weather data CSV file
        """
        self.file_path = file_path
        self.df = None
        self.cleaned_df = None
        self.summary_stats = {}
        
    def task1_load_data(self):
        """
        Task 1: Data Acquisition and Loading
        Load CSV file into Pandas DataFrame and inspect its structure.
        """
        print("=" * 50)
        print("TASK 1: DATA LOADING AND INSPECTION")
        print("=" * 50)
        
        try:
            # Load the CSV file
            self.df = pd.read_csv(self.file_path)
            print(f"✓ Data loaded successfully. Shape: {self.df.shape}")
            
            # Display basic information
            print("\n1. First 5 rows:")
            print(self.df.head())
            
            print("\n2. DataFrame Info:")
            print(self.df.info())
            
            print("\n3. Statistical Summary:")
            print(self.df.describe())
            
            print("\n4. Columns in dataset:")
            print(list(self.df.columns))
            
            return True
        except FileNotFoundError:
            print(f"✗ Error: File not found at {self.file_path}")
            return False
        except Exception as e:
            print(f"✗ Error loading file: {e}")
            return False
    
    def task2_clean_data(self):
        """
        Task 2: Data Cleaning and Processing
        Handle missing values, convert date columns, and filter relevant columns.
        """
        print("\n" + "=" * 50)
        print("TASK 2: DATA CLEANING AND PROCESSING")
        print("=" * 50)
        
        if self.df is None:
            print("✗ Error: No data loaded. Run task1_load_data() first.")
            return False
        
        # Create a copy for cleaning
        self.cleaned_df = self.df.copy()
        
        # 1. Handle missing values
        print("\n1. Handling Missing Values:")
        missing_before = self.cleaned_df.isnull().sum().sum()
        print(f"   Total missing values before cleaning: {missing_before}")
        
        # Fill numeric columns with mean, drop rows with too many missing values
        numeric_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.cleaned_df[col].isnull().any():
                self.cleaned_df[col].fillna(self.cleaned_df[col].mean(), inplace=True)
        
        # Drop remaining rows with any missing values
        self.cleaned_df.dropna(inplace=True)
        
        missing_after = self.cleaned_df.isnull().sum().sum()
        print(f"   Total missing values after cleaning: {missing_after}")
        
        # 2. Convert date columns to datetime
        print("\n2. Converting Date Columns:")
        date_columns = [col for col in self.cleaned_df.columns 
                       if 'date' in col.lower() or 'time' in col.lower()]
        
        if date_columns:
            for date_col in date_columns:
                try:
                    self.cleaned_df[date_col] = pd.to_datetime(self.cleaned_df[date_col])
                    print(f"   ✓ Converted '{date_col}' to datetime")
                except:
                    print(f"   ✗ Could not convert '{date_col}' to datetime")
        else:
            print("   No date columns found. Looking for common date column names...")
            # Try to find columns that might contain dates
            for col in self.cleaned_df.columns:
                if any(keyword in col.lower() for keyword in ['year', 'month', 'day']):
                    try:
                        self.cleaned_df[col] = pd.to_datetime(self.cleaned_df[col])
                        print(f"   ✓ Converted '{col}' to datetime")
                    except:
                        continue
        
        # 3. Filter relevant columns
        print("\n3. Filtering Relevant Columns:")
        relevant_keywords = ['temp', 'rain', 'humid', 'pressure', 'wind']
        relevant_cols = []
        
        for col in self.cleaned_df.columns:
            if any(keyword in col.lower() for keyword in relevant_keywords):
                relevant_cols.append(col)
        
        # Keep date columns and relevant weather columns
        date_cols = [col for col in self.cleaned_df.columns 
                    if pd.api.types.is_datetime64_any_dtype(self.cleaned_df[col])]
        
        all_relevant_cols = date_cols + relevant_cols
        
        if all_relevant_cols:
            self.cleaned_df = self.cleaned_df[all_relevant_cols]
            print(f"   Kept {len(all_relevant_cols)} relevant columns:")
            for col in all_relevant_cols:
                print(f"     - {col}")
        else:
            print("   No relevant columns found. Keeping all columns.")
        
        print(f"\n✓ Data cleaning complete. New shape: {self.cleaned_df.shape}")
        return True
    
    def task3_statistical_analysis(self):
        """
        Task 3: Statistical Analysis with NumPy
        Compute daily, monthly, and yearly statistics.
        """
        print("\n" + "=" * 50)
        print("TASK 3: STATISTICAL ANALYSIS")
        print("=" * 50)
        
        if self.cleaned_df is None:
            print("✗ Error: No cleaned data available. Run task2_clean_data() first.")
            return False
        
        # Find date column
        date_col = None
        for col in self.cleaned_df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.cleaned_df[col]):
                date_col = col
                break
        
        if date_col is None:
            print("✗ Error: No datetime column found for analysis")
            return False
        
        # Set date as index for resampling
        temp_df = self.cleaned_df.copy()
        temp_df.set_index(date_col, inplace=True)
        
        # Find numeric columns for analysis
        numeric_cols = temp_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("✗ Error: No numeric columns found for analysis")
            return False
        
        print("\n1. Overall Statistics (Using NumPy):")
        for col in numeric_cols:
            data = temp_df[col].values
            self.summary_stats[col] = {
                'mean': np.mean(data),
                'min': np.min(data),
                'max': np.max(data),
                'std': np.std(data),
                'median': np.median(data)
            }
            print(f"\n   {col.upper()}:")
            print(f"     Mean: {np.mean(data):.2f}")
            print(f"     Min: {np.min(data):.2f}")
            print(f"     Max: {np.max(data):.2f}")
            print(f"     Std Dev: {np.std(data):.2f}")
            print(f"     Median: {np.median(data):.2f}")
        
        # Daily statistics
        print("\n2. Daily Statistics:")
        try:
            daily_stats = temp_df[numeric_cols].resample('D').agg(['mean', 'min', 'max', 'std'])
            print(f"   ✓ Computed daily statistics for {len(daily_stats)} days")
            print(f"   Sample daily mean for first 5 days:")
            print(daily_stats.head())
        except Exception as e:
            print(f"   ✗ Error computing daily statistics: {e}")
        
        # Monthly statistics
        print("\n3. Monthly Statistics:")
        try:
            monthly_stats = temp_df[numeric_cols].resample('M').agg(['mean', 'min', 'max', 'std'])
            print(f"   ✓ Computed monthly statistics for {len(monthly_stats)} months")
            print(f"   Sample monthly mean for first 3 months:")
            print(monthly_stats.head(3))
        except Exception as e:
            print(f"   ✗ Error computing monthly statistics: {e}")
        
        # Yearly statistics
        print("\n4. Yearly Statistics:")
        try:
            yearly_stats = temp_df[numeric_cols].resample('Y').agg(['mean', 'min', 'max', 'std'])
            print(f"   ✓ Computed yearly statistics for {len(yearly_stats)} years")
            print(f"   Yearly statistics:")
            print(yearly_stats)
        except Exception as e:
            print(f"   ✗ Error computing yearly statistics: {e}")
        
        return True
    
    def task5_grouping_aggregation(self):
        """
        Task 5: Grouping and Aggregation
        Group data by month or season and calculate aggregate statistics.
        """
        print("\n" + "=" * 50)
        print("TASK 5: GROUPING AND AGGREGATION")
        print("=" * 50)
        
        if self.cleaned_df is None:
            print("✗ Error: No cleaned data available. Run task2_clean_data() first.")
            return False
        
        # Find date column
        date_col = None
        for col in self.cleaned_df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.cleaned_df[col]):
                date_col = col
                break
        
        if date_col is None:
            print("✗ Error: No datetime column found for grouping")
            return False
        
        # Create a copy for grouping operations
        group_df = self.cleaned_df.copy()
        
        # Extract month and year for grouping
        group_df['year'] = group_df[date_col].dt.year
        group_df['month'] = group_df[date_col].dt.month
        group_df['season'] = group_df[date_col].dt.month % 12 // 3 + 1
        
        # Map season numbers to names
        season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
        group_df['season_name'] = group_df['season'].map(season_map)
        
        # Find numeric columns
        numeric_cols = group_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['year', 'month', 'season']]
        
        print("\n1. Grouping by Month:")
        monthly_groups = group_df.groupby('month')[numeric_cols].agg(['mean', 'min', 'max', 'std'])
        print(monthly_groups)
        
        print("\n2. Grouping by Season:")
        seasonal_groups = group_df.groupby('season_name')[numeric_cols].agg(['mean', 'min', 'max', 'std'])
        print(seasonal_groups)
        
        print("\n3. Grouping by Year:")
        yearly_groups = group_df.groupby('year')[numeric_cols].agg(['mean', 'min', 'max', 'std'])
        print(yearly_groups)
        
        return True
    
    def task6_export_data(self):
        """
        Task 6: Export and Storytelling
        Export cleaned data to CSV and generate summary report.
        """
        print("\n" + "=" * 50)
        print("TASK 6: EXPORT AND REPORTING")
        print("=" * 50)
        
        if self.cleaned_df is None:
            print("✗ Error: No cleaned data available. Run task2_clean_data() first.")
            return False
        
        # Create output directory
        output_dir = "weather_analysis_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Export cleaned data to CSV
        cleaned_file_path = os.path.join(output_dir, "cleaned_weather_data.csv")
        self.cleaned_df.to_csv(cleaned_file_path, index=False)
        print(f"✓ Cleaned data exported to: {cleaned_file_path}")
        
        # 2. Create summary statistics file
        stats_file_path = os.path.join(output_dir, "weather_statistics.txt")
        with open(stats_file_path, 'w') as f:
            f.write("WEATHER DATA STATISTICAL SUMMARY\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("Dataset Information:\n")
            f.write(f"- Original shape: {self.df.shape}\n")
            f.write(f"- Cleaned shape: {self.cleaned_df.shape}\n")
            f.write(f"- Columns in cleaned data: {list(self.cleaned_df.columns)}\n\n")
            
            f.write("Statistical Summary:\n")
            for col, stats in self.summary_stats.items():
                f.write(f"\n{col.upper()}:\n")
                for stat_name, value in stats.items():
                    f.write(f"  {stat_name.title()}: {value:.2f}\n")
        
        print(f"✓ Statistics summary exported to: {stats_file_path}")
        
        # 3. Create detailed report in Markdown format
        report_file_path = os.path.join(output_dir, "weather_analysis_report.md")
        
        # Find date column for time-based insights
        date_col = None
        for col in self.cleaned_df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.cleaned_df[col]):
                date_col = col
                break
        
        with open(report_file_path, 'w') as f:
            f.write("# Weather Data Analysis Report\n\n")
            
            f.write("## 1. Executive Summary\n")
            f.write("This report summarizes the analysis of weather data ")
            f.write("including data cleaning, statistical analysis, and key insights.\n\n")
            
            f.write("## 2. Dataset Overview\n")
            f.write(f"- **Original dataset size**: {self.df.shape[0]} rows × {self.df.shape[1]} columns\n")
            f.write(f"- **Cleaned dataset size**: {self.cleaned_df.shape[0]} rows × {self.cleaned_df.shape[1]} columns\n")
            
            if date_col:
                date_range = self.cleaned_df[date_col]
                f.write(f"- **Time period**: {date_range.min().date()} to {date_range.max().date()}\n")
            
            f.write(f"- **Missing values handled**: Yes (filled with mean values)\n\n")
            
            f.write("## 3. Key Statistics\n")
            for col, stats in self.summary_stats.items():
                f.write(f"\n### {col.replace('_', ' ').title()}\n")
                f.write(f"- **Average**: {stats['mean']:.2f}\n")
                f.write(f"- **Range**: {stats['min']:.2f} to {stats['max']:.2f}\n")
                f.write(f"- **Variability (std)**: {stats['std']:.2f}\n")
                f.write(f"- **Median**: {stats['median']:.2f}\n")
            
            f.write("\n## 4. Insights and Observations\n")
            f.write("1. **Data Quality**: The dataset required cleaning for missing values.\n")
            f.write("2. **Temperature Patterns**: [Add specific observations based on your data]\n")
            f.write("3. **Rainfall Distribution**: [Add specific observations based on your data]\n")
            f.write("4. **Seasonal Variations**: [Add specific observations based on your data]\n\n")
            
            f.write("## 5. Recommendations for Campus Sustainability\n")
            f.write("1. Based on temperature trends, consider optimizing HVAC usage\n")
            f.write("2. Rainfall patterns can inform water conservation strategies\n")
            f.write("3. Extreme weather events should be monitored for campus safety\n\n")
            
            f.write("## 6. Files Generated\n")
            f.write(f"- `{cleaned_file_path}`: Cleaned dataset\n")
            f.write(f"- `{stats_file_path}`: Statistical summary\n")
            f.write(f"- `{report_file_path}`: This report\n")
        
        print(f"✓ Detailed report exported to: {report_file_path}")
        print(f"\n✓ All files saved in directory: {output_dir}/")
        
        return True
    
    def run_all_tasks(self):
        """Execute all tasks in sequence."""
        print("STARTING WEATHER DATA ANALYSIS")
        print("=" * 50)
        
        tasks = [
            ("Loading Data", self.task1_load_data),
            ("Cleaning Data", self.task2_clean_data),
            ("Statistical Analysis", self.task3_statistical_analysis),
            ("Grouping and Aggregation", self.task5_grouping_aggregation),
            ("Export and Reporting", self.task6_export_data)
        ]
        
        for task_name, task_func in tasks:
            print(f"\n▶ Starting: {task_name}")
            success = task_func()
            if not success:
                print(f"✗ Task '{task_name}' failed. Stopping analysis.")
                return False
        
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE!")
        print("=" * 50)
        print("\nSummary of generated files:")
        print("- cleaned_weather_data.csv: Cleaned dataset")
        print("- weather_statistics.txt: Statistical summary")
        print("- weather_analysis_report.md: Detailed analysis report")
        print("\nNote: Visualization code was omitted as requested.")
        return True


# Main execution
if __name__ == "__main__":
    # Example usage - replace with your actual file path
    file_path = "weather_data.csv"  # Change this to your actual file path
    
    # Create analyzer instance
    analyzer = WeatherDataAnalyzer(file_path)
    
    # Run all tasks
    analyzer.run_all_tasks()
    
    # For testing with sample data if no file is available
    # Uncomment the following lines to create sample data:
    """
    import pandas as pd
    import numpy as np
    
    # Create sample weather data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    sample_data = {
        'date': dates,
        'temperature': np.random.normal(25, 5, len(dates)),
        'rainfall': np.random.exponential(2, len(dates)),
        'humidity': np.random.uniform(40, 90, len(dates)),
        'wind_speed': np.random.uniform(0, 15, len(dates))
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_weather_data.csv', index=False)
    print("Sample data created: sample_weather_data.csv")
    """