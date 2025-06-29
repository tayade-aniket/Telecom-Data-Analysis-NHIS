# TellCo Telecom Data Analysis 📊📈📱

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![SQL Server](https://img.shields.io/badge/Microsoft%20SQL%20Server-CC2927?style=for-the-badge&logo=microsoft%20sql%20server&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 🌟 Project Overview

This project delivers a comprehensive data analysis of **TellCo's telecom dataset**, providing critical insights to inform a strategic investment decision. As an analyst for a wealthy investor, the goal is to uncover opportunities for growth and assess TellCo's overall value by examining customer behavior and network performance.

The diagram below illustrates the high-level data flow and analytical process:

<img src="/assets/steps.svg" alt="Data Flow Diagram" width="250" height="450" />

## 🔍 Detailed Analysis Sections

### 1. User Engagement Analysis
**Objective:** Identify user engagement patterns based on session frequency, duration, and data usage.

**Key Steps:**
- Cleaned and normalized session duration and data usage metrics
- Performed K-Means clustering to segment users into engagement groups
- Identified top users by data consumption
- Stored results in SQL database for further analysis

<img src="/assets/user_engement_plot.png" alt="Data Flow Diagram" width="750" height="400" />

**Insights:**
- Clear segmentation of users into low, medium, and high engagement clusters
- Identified power users consuming disproportionate network resources
- Established baseline metrics for normal user behavior

<img src="/assets/1 user_engagement_analysis.png" alt="Data Flow Diagram" width="750" height="400" />

### 2. User Experience Analysis
**Objective:** Evaluate network performance metrics affecting user experience.

**Key Steps:**
- Analyzed critical network parameters (RTT, throughput, TCP retransmissions)
- Implemented outlier detection using z-score normalization
- Created experience clusters using K-Means
- Flagged users with poor network performance

<img src="/assets/user_experience_plot.png" alt="Data Flow Diagram" width="750" height="400" />

**Insights:**
- Identified users experiencing suboptimal network conditions
- Established correlation between throughput and RTT
- Highlighted areas needing network infrastructure improvements

<img src="/assets/2 user_experience_analysis.png" alt="Data Flow Diagram" width="750" height="400" />

### 3. User Satisfaction Analysis
**Objective:** Develop a composite satisfaction score combining engagement and experience metrics.

**Key Steps:**
- Merged engagement and experience datasets
- Created weighted satisfaction score (40% data usage, 30% RTT, 30% throughput)
- Performed regression analysis to validate score components
- Identified most and least satisfied users

<img src="/assets/user_satisfaction_plot.png" alt="Data Flow Diagram" width="750" height="400" />

**Insights:**
- Strong correlation between throughput and satisfaction
- High-engagement users not always most satisfied
- Identified at-risk users for targeted retention programs

<img src="/assets/3 user_satisfaction_analysis.png" alt="Data Flow Diagram" width="750" height="400" />

## 🎯 Business Context & Objective

The investor specializes in acquiring undervalued assets and relies on detailed data analysis to understand business fundamentals and drive profitability. TellCo, a mobile service provider in the Republic of Pefkakia, has shared financial data but lacks insights from its system-generated data.

**Objective:** To analyze growth opportunities and recommend whether TellCo is a worthy acquisition, leveraging a detailed telecommunication dataset.

## 💾 Dataset

The analysis is based on a simulated **telecom xDR (data sessions Detail Record)** dataset, encompassing aggregated data from one month. It provides rich information on:

- **User Behavior:** Tracking activities across popular applications like Social Media, Google, Email, YouTube, Netflix, Gaming, and Others.
- **Network Performance:** Key metrics such as TCP retransmission, Round Trip Time (RTT), and Throughput.
- **Device Information:** Details on Handset Manufacturer and Handset Type.

## 🛠️ Technical Stack

- **Language:** Python 🐍, SQL (T-SQL)
- **Database:** SQL Server
- **Data Manipulation:** pandas, numpy
- **Data Visualization:** matplotlib, seaborn, plotly
- **Machine Learning:** scikit-learn (for K-Means clustering, PCA)
- **Statistical Analysis:** scipy

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/tayade-aniket/Telecom-Data-Analysis-NHIS
cd Telecom-Data-Analysis-NHIS

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook
```

## 📊 Analysis Workflow

<img src="/assets/diagram.png" alt="Data Flow Diagram" width="450" height="500" />

# ✨Investment Recommendation

### Growth Potential Assessment

- High engagement users are already using a lot of data, especially for video — they can be upsold premium plans or features.
- Platforms like YouTube and Netflix show users want fast, high-bandwidth connections, making a strong case for 5G upgrades.
- Many users own top-end phones like Samsung and Apple, which shows they’re willing to spend — a good sign for boosting ARPU.
- Partnering with device brands could be a smart way to sell bundled services or offers, boosting both device and plan sales.
- The low-usage group still has potential — with the right offers or improvements, they could become more engaged and loyal.
- Users in the "Poor Experience" group may leave if not addressed — improving their network can reduce churn and win trust.
- There's a strong business case to improve service quality — it directly ties to happier users and higher revenue.
- Overall, by targeting both top users and struggling users, the company can achieve balanced growth and retention.

### Investment Recommendation: **BUY**

TellCo appears to be a viable acquisition target with significant growth potential, provided the investor is prepared to address existing network experience issues and capitalize on identified opportunities.

### Limitations and Risks

* The data is simulated, so real-world results could be very different due to missing variables like age, income, or actual churn behavior.
* The network shows technical issues like high RTT and retransmissions — fixing this will need big investments in infrastructure.
* Since churn data is missing, the current satisfaction score is more of a guess than a proven churn indicator.
* The competitive environment isn’t analyzed, meaning TellCo's position in the market isn’t fully clear.
* Local telecom laws and regulations in Pefkakia may affect costs, expansion, and service — but they’re not considered here.
* Economic and political conditions in the country can greatly impact TellCo’s growth, but these aren't part of the analysis.
* Overall, while the insights are useful, real business decisions need deeper, real-world data and context for accuracy.

### Key Strategic Recommendations for Post-Acquisition

- **Fix the network first** — high delay (RTT) and retransmissions are hurting user experience, so infrastructure upgrades are urgent.
- **Launch custom data plans** — like unlimited video for heavy users, and small trial packs for low users, to boost engagement.
- **Set up a churn alert system** — reach out to users with bad experience scores using offers, support, or incentives to retain them.
- **Collaborate with top handset brands (like Apple and Samsung)** — to sell bundled offers and ensure device–network optimization.
- **Build a strong analytics** — team to keep track of usage, satisfaction, and network data in real-time for faster decisions.
- **Use app usage patterns (e.g., high YouTube demand)** — to guide service expansion and improve coverage in high-potential areas.
- **Final note:** While this analysis gives a good direction, real-world data and competitive research are crucial before investing.

## Authors

[@tayade-aniket](https://github.com/tayade-aniket)