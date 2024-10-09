# A/B Test Analyzer

A Python tool for comprehensive A/B testing analysis. This tool provides summary statistics, visualizations, and statistical tests to evaluate differences between control and treatment groups.

## Features

- Summarizes statistics for control and treatment groups
- Visualizes data using histograms and box plots
- Performs normality tests using Shapiro-Wilk test
- Tests for homogeneity of variances using Levene's test
- Conducts independent samples t-tests and Mann-Whitney U tests
  
## Usage

1. Import the necessary libraries and the `ABTestAnalyzer` class:

    ```python
    import pandas as pd
    import numpy as np
    from ab_test_analyzer import ABTestAnalyzer
    ```

2. Load your data into a pandas DataFrame:
    ```python
    data = pd.read_csv('your_data_file.csv')
    ```

3. Create an instance of the `ABTestAnalyzer` class:
    ```python
    ab_test_analyzer = ABTestAnalyzer(data)
    ```

4. Analyze a specific variable:
    ```python
    ab_test_analyzer.analyze_variable('your_variable')
    ```
