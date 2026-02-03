
---

# **PropertyMarket**

## **Overview**

This repository contains a **cleaned, structured dataset of residential property sales** collected from Redfin for **four ZIP codes**. The dataset is suitable for **data analysis, visualization, and machine learning projects** related to the real estate market.

It includes key property features such as **sale price, sale date, property type, bedrooms, bathrooms, square footage, and year built**. All data has been normalized and missing values handled consistently to ensure reliability and usability.

---

## **Dataset Scope**

* **Source:** Redfin ([www.redfin.com](http://www.redfin.com))
* **Geographic Coverage:** 4 ZIP codes (user-defined)
* **Timeframe:** Most recent listings and historical sales available via Redfin pages
* **Number of Records:** Variable; includes all listings across the selected ZIP codes
* **Data Features:**

| Feature        | Description                           |
| -------------- | ------------------------------------- |
| ZIP            | ZIP code of the property              |
| Address        | Street address                        |
| Sale Price     | Final sale price (USD)                |
| Sale Date      | Date of sale (YYYY-MM-DD)             |
| Property Type  | Single Family, Condo, Townhouse, etc. |
| Beds           | Number of bedrooms                    |
| Baths          | Number of bathrooms                   |
| SqFt           | Living area in square feet            |
| Year Built     | Year the property was built           |
| Price per SqFt | Computed price per square foot        |

---

## **Project Structure**

```
Redfin-House-Sales-4ZIPs/
│
├── redfin_4zip_scraper.py        # Python scraper with pagination and cleaning
├── redfin_4zip_dataset_clean.csv # Final cleaned dataset
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies
```

---

## **Data Collection Methodology**

1. **Scraping Redfin:**

   * The project uses Python (`requests` + `BeautifulSoup`) to extract data from Redfin ZIP code pages.
   * Pagination is implemented to scrape **all listings** per ZIP code.

2. **Data Cleaning and Normalization:**

   * Numeric features (`Sale Price`, `Beds`, `Baths`, `SqFt`, `Year Built`) are converted and missing values handled.
   * Missing dates are replaced with a placeholder (`1900-01-01`).
   * Property types are standardized and missing entries replaced with `"Unknown"`.
   * `Price per SqFt` is calculated automatically.

3. **Ethical Considerations:**

   * Scraping is performed **responsibly** for academic purposes.
   * Rate limiting (`time.sleep`) is used to avoid overloading the Redfin servers.
   * No personal or sensitive information is collected.

---

## **Dependencies**

* Python 3.9+
* `requests`
* `beautifulsoup4`
* `pandas`

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## **Usage**

1. **Scrape Data (Optional):**

   ```bash
   python redfin_4zip_scraper.py
   ```

   * Outputs a cleaned CSV (`redfin_4zip_dataset_clean.csv`).

2. **Load Dataset in Python:**

   ```python
   import pandas as pd

   df = pd.read_csv("redfin_4zip_dataset_clean.csv")
   print(df.head())
   ```

3. **SQL Import (Optional):**

   * The dataset can be loaded into MySQL, PostgreSQL, or SQLite for queries and analysis.

---

## **Potential Applications**

* Real estate price analysis by ZIP code
* Predictive modeling (house price prediction)
* Comparative market studies
* Data visualization (heatmaps, trends, histograms)

---

## **License**

This project is for **educational and research purposes**. Data is scraped from publicly available Redfin listings and intended for **non-commercial use**.

---


