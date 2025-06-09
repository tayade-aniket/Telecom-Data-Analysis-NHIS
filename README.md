# TellCo Telecom Data Analysis

## Project Overview

This project provides a comprehensive data analysis of TellCo's telecom dataset. The primary objective is to support an investment decision regarding TellCo, an existing mobile service provider. The analysis delves into various aspects of user behavior, engagement, experience, and satisfaction to identify growth opportunities, assess network performance, and segment customers effectively.

## Features

-   **User Overview Analysis:** Initial exploration of the dataset to understand its structure, identify missing values, and gain basic insights into the data.
-   **Handset Analysis:** Identifies the most popular handsets and manufacturers, and analyzes their impact on data usage and session duration.
-   **Application Usage Analysis:** Breaks down data consumption by various applications (Social Media, Google, Email, YouTube, Netflix, Gaming, Others) and identifies key usage patterns.
-   **Statistical Analysis:** Provides descriptive statistics, distribution analysis, and correlation insights for key metrics.
-   **User Engagement Analysis:** Segments users into different engagement clusters (e.g., low, medium, high) based on session frequency, duration, and data volume using K-Means clustering.
-   **User Experience Analysis:** Segments users based on network performance metrics (Average Round Trip Time (RTT), Throughput, TCP Retransmissions) to identify users experiencing poor service quality.
-   **User Satisfaction Analysis:** Develops a composite satisfaction score by combining engagement and experience metrics, helping to identify both highly satisfied and at-risk (churn) users.
-   **Investment Recommendation:** Provides a "Buy/Don't Buy" recommendation for TellCo, supported by an assessment of growth potential, identified limitations, and strategic recommendations.

## Getting Started

To run this analysis locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/TellCo-Telecom-Data-Analysis.git](https://github.com/YourUsername/TellCo-Telecom-Data-Analysis.git)
    cd TellCo-Telecom-Data-Analysis
    ```
2.  **Install dependencies:**
    Ensure you have Python installed. Then, install the required libraries using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn plotly scikit-learn scipy
    ```
3.  **Run the Jupyter Notebook:**
    Open the `TellCo_Telecom_Analysis.ipynb` file in a Jupyter environment (Jupyter Notebook or JupyterLab) and run all cells.
    ```bash
    jupyter notebook TellCo_Telecom_Analysis.ipynb
    ```

## Dataset

The analysis uses a simulated telecom xDR (data sessions Detail Record) dataset. It includes information on:
-   User behavior across applications (Social Media, Google, Email, YouTube, Netflix, Gaming, Others)
-   Network performance metrics (TCP retransmission, RTT, Throughput)
-   Device information (Handset Manufacturer, Handset Type)
-   Session details (duration, start/end times)

## Insights and Recommendations

The project provides actionable insights into customer segmentation, network performance bottlenecks, and application-specific usage patterns. The final investment recommendation highlights potential growth areas and outlines strategic steps for improving customer satisfaction and retention.

---

📊📈📱