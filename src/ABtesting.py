import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
import statistics
from termcolor import colored
import numpy as np
import pandas as pd


class ABTestAnalyzer:
    def __init__(self, data, control_group_label='control', treatment_group_label='treatment'):
        self.data = data
        self.control_group_label = control_group_label
        self.treatment_group_label = treatment_group_label
        self.control_group = data[data["group"] == control_group_label]
        self.treatment_group = data[data["group"] == treatment_group_label]

    def summary_statistics(self, variable):
        print(colored(f"Summary Statistics by Groups for {variable}", "cyan", attrs=["bold"]), "\n")
        stats = self.data.groupby("group")[variable].aggregate(["count", "mean", "std", "median", "min", "max"])
        print(stats.transpose(), "\n")

    def plot_histograms(self, variable):
        print(colored(f"Histogram by Groups for {variable}", "cyan", attrs=["bold"]), "\n")
        sns.histplot(self.control_group[variable], color="skyblue", label="Control Group", kde=True)
        sns.histplot(self.treatment_group[variable], color="red", label="Treatment Group", kde=True)
        plt.legend()
        plt.show()

    def plot_boxplots(self, variable):
        self.data.boxplot(column=[variable], by="group", return_type=None)
        plt.suptitle("")
        print(colored(f"Box Plot by Groups for {variable}", "cyan", attrs=["bold"]), "\n")
        plt.show()

    def normality_test(self, variable):
        group_control_stat, group_control_p = shapiro(self.control_group[variable])
        group_treatment_stat, group_treatment_p = shapiro(self.treatment_group[variable])

        print(colored(f"1. Step: Testing the Normality Assumption for {variable}", "cyan", attrs=["bold"]), "\n")
        print(f"control_group Shapiro-Wilk p-value = {group_control_p:.3f}, treatment_group "
              f"Shapiro-Wilk p-value = {group_treatment_p:.3f}\n")

        if group_control_p > 0.05 and group_treatment_p > 0.05:
            print(f"Shapiro-Wilk Test: p > .05 for both groups; the distribution of {variable} is likely normal.\n")
            return True
        elif group_control_p < 0.05 and group_treatment_p < 0.05:
            print(f"Shapiro-Wilk Test: p < .05 for both groups; the distribution of {variable} is not normal.\n")
            return False
        elif group_control_p > 0.05:
            print(f"Shapiro-Wilk Test: p > .05 for control_group and p < .05 for treatment_group; "
                  f"check for outliers in treatment_group.\n")
            return False
        else:
            print(f"Shapiro-Wilk Test: p > .05 for treatment_group and p < .05 for control_group; "
                  f"check for outliers in control_group.\n")
            return False

    def homogeneity_test(self, variable):
        stat, p_value = levene(self.control_group[variable], self.treatment_group[variable])

        print(colored(f"2. Step: Testing the Homogeneity Assumption for {variable}", "cyan", attrs=["bold"]), "\n")
        print(f"Levene's Test: stat = {stat:.3f}, p-value = {p_value:.3f}\n")

        if p_value > 0.05:
            print("Levene's Test: p > .05; variances are equal.\n")
            return True
        else:
            print("Levene's Test: p < .05; variances are not equal.\n")
            return False

    def independent_samples_t_test(self, variable, equal_var=True):
        t_stat, p_value = ttest_ind(self.control_group[variable], self.treatment_group[variable], equal_var=equal_var)

        print(colored(f"3. Step: Independent Samples t Test for {variable}", "cyan", attrs=["bold"]), "\n")
        print(f"t-test: t-stat = {t_stat:.3f}, p-value = {p_value:.3f}\n")

        if p_value > 0.05:
            print(f"t-test: p > .05; no significant difference in {variable} between the groups.\n")
        else:
            print(f"t-test: p < .05; significant difference in {variable} between the groups.\n")
            if statistics.mean(self.control_group[variable]) > statistics.mean(self.treatment_group[variable]):
                print(f"Mean of control_group in {variable} is greater than treatment_group\n")
            else:
                print(f"Mean of treatment_group in {variable} is greater than control_group\n")

    def mann_whitney_u_test(self, variable):
        u_stat, p_value = mannwhitneyu(self.control_group[variable], self.treatment_group[variable])

        print(colored(f"3. Step: Mann-Whitney U Test for {variable}", "cyan", attrs=["bold"]), "\n")
        print(f"Mann-Whitney U Test: u-stat = {u_stat:.3f}, p-value = {p_value:.3f}\n")

        if p_value > 0.05:
            print(
                f"Mann-Whitney U Test: p > .05; no significant difference in {variable} distributions between the groups.\n")
        else:
            print(
                f"Mann-Whitney U Test: p < .05; significant difference in {variable} distributions between the groups.\n")
            if statistics.median(self.control_group[variable]) > statistics.median(self.treatment_group[variable]):
                print(f"Median of control_group in {variable} is greater than treatment_group\n")
            else:
                print(f"Median of treatment_group in {variable} is greater than control_group\n")

    def remove_outliers_iqr(self, group, column):
        Q1 = group[column].quantile(0.25)
        Q3 = group[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

        filtered_group = group[(group[column] >= lower_bound) & (group[column] <= upper_bound)]

        if group.equals(self.control_group):
            self.control_group = filtered_group
        else:
            self.treatment_group = filtered_group

    def remove_outliers_zscore(self, group, column, threshold=3):
        mean = np.mean(group[column])
        std = np.std(group[column])
        group['z_score'] = (group[column] - mean) / std
        group_filtered = group[group['z_score'].abs() <= threshold]
        group_filtered = group_filtered.drop(columns=['z_score'])
        if group.equals(self.control_group):
            self.control_group = group_filtered
        else:
            self.treatment_group = group_filtered

    def analyze_variable(self, variable):
        print(colored(f"A/B Testing for {variable}", "cyan", attrs=["bold", 'reverse', 'blink']), "\n")

        # Update the combined data after outlier removal
        self.data = pd.concat([self.control_group, self.treatment_group])

        self.summary_statistics(variable)

        self.plot_histograms(variable)
        self.plot_boxplots(variable)

        normal = self.normality_test(variable)

        if normal:
            equal_var = self.homogeneity_test(variable)
            self.independent_samples_t_test(variable, equal_var=equal_var)
        else:
            self.mann_whitney_u_test(variable)
