
# Import the Necessary Libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from urllib.parse import urlencode
import json


class HouseDataCollector:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.data = []

    def scrape_redfin(self, zip_code, max_pages=3):

        # Scrape house data from Redfin for a given zip code

        print(f"\nScraping Redfin for zip code: {zip_code}")

        for page in range(1, max_pages + 1):
            try:
                # Redfin URL structure
                url = f"https://www.redfin.com/zipcode/{zip_code}/filter/include=sold-3mo"
                if page > 1:
                    url += f"/page-{page}"

                print(f"  Page {page}: {url}")

                response = requests.get(url, headers=self.headers, timeout=10)
                time.sleep(random.uniform(2, 4))  # Random delay to avoid detection

                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Find property cards
                    properties = soup.find_all('div', class_='HomeCardContainer')

                    if not properties:
                        # Try alternative selector
                        properties = soup.find_all('div', attrs={'data-rf-test-name': 'propertyCard'})

                    print(f"Found {len(properties)} properties")

                    for prop in properties:
                        try:
                            house_data = self.parse_redfin_property(prop, zip_code)
                            if house_data:
                                self.data.append(house_data)
                        except Exception as e:
                            print(f"Error parsing property: {e}")
                            continue

                else:
                    print(f"Failed to fetch page. Status code: {response.status_code}")

            except Exception as e:
                print(f"  Error on page {page}: {e}")
                continue

        return len(self.data)

    def parse_redfin_property(self, property_element, zip_code):

        # Parse individual property data from Redfin
        try:
            # Extract beds
            beds_elem = property_element.find('div', class_='bp-Homecard__Stats--beds')
            if not beds_elem:
                beds_elem = property_element.find('span', attrs={'data-rf-test-id': 'property-beds'})
            beds = self.extract_number(beds_elem.text if beds_elem else None)

            # Extract baths
            baths_elem = property_element.find('div', class_='bp-Homecard__Stats--baths')
            if not baths_elem:
                baths_elem = property_element.find('span', attrs={'data-rf-test-id': 'property-baths'})
            baths = self.extract_number(baths_elem.text if baths_elem else None)

            # Extract sqft
            sqft_elem = property_element.find('div', class_='bp-Homecard__Stats--sqft')
            if not sqft_elem:
                sqft_elem = property_element.find('span', attrs={'data-rf-test-id': 'property-sqft'})
            sqft = self.extract_number(sqft_elem.text if sqft_elem else None)

            # Extract price
            price_elem = property_element.find('span', class_='bp-Homecard__Price--value')
            if not price_elem:
                price_elem = property_element.find('div', attrs={'data-rf-test-id': 'property-price'})
            price = self.extract_number(price_elem.text if price_elem else None)

            if all([beds, baths, sqft, price]):
                return {
                    'beds': beds,
                    'baths': baths,
                    'sqft': sqft,
                    'price': price,
                    'zip_code': zip_code
                }
        except Exception as e:
            print(f"Parse error: {e}")
            return None

    def extract_number(self, text):

        # Extract numeric value from text
        if not text:
            return None

        # Remove non-numeric characters except dots
        import re
        numbers = re.findall(r'[\d.]+', text.replace(',', ''))
        if numbers:
            try:
                return float(numbers[0])
            except:
                return None
        return None

    def generate_synthetic_data(self, zip_codes):
        """
        Generate synthetic house data based on realistic distributions
        This is a fallback if web scraping doesn't work due to website changes
        """
        print("\n=== Generating Synthetic Dataset ===")
        print("(Using realistic distributions for demonstration)")

        import numpy as np
        np.random.seed(42)

        # Define characteristics for each zip code
        zip_profiles = {
            '07008': {'price_mean': 550000, 'price_std': 150000, 'sqft_mean': 2200, 'sqft_std': 600},
            '10001': {'price_mean': 1200000, 'price_std': 400000, 'sqft_mean': 1400, 'sqft_std': 400},
            '19038': {'price_mean': 450000, 'price_std': 120000, 'sqft_mean': 2000, 'sqft_std': 500},
            '95001': {'price_mean': 900000, 'price_std': 300000, 'sqft_mean': 1800, 'sqft_std': 500}
        }

        samples_per_zip = 75  # 75 samples per zip code = 300 total

        for zip_code in zip_codes:
            profile = zip_profiles[zip_code]

            for _ in range(samples_per_zip):
                # Generate correlated features
                sqft = max(500, np.random.normal(profile['sqft_mean'], profile['sqft_std']))

                # Beds correlate with sqft
                beds = max(1, min(6, int(sqft / 600 + np.random.normal(0, 0.5))))

                # Baths correlate with beds and sqft
                baths = max(1, min(5, beds - np.random.choice([0, 1]) + np.random.normal(0, 0.3)))
                baths = round(baths * 2) / 2  # Round to nearest 0.5

                # Price correlates with sqft and location
                base_price = sqft * (profile['price_mean'] / profile['sqft_mean'])
                price = max(100000, np.random.normal(base_price, profile['price_std']))

                self.data.append({
                    'beds': int(beds),
                    'baths': float(baths),
                    'sqft': int(sqft),
                    'price': int(price),
                    'zip_code': zip_code
                })

    def save_to_csv(self, filename='house_data.csv'):

        # Save collected data to CSV

        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)
        print(f"\n=== Data saved to {filename} ===")
        print(f"Total records: {len(df)}")
        print(f"\nDataset preview:")
        print(df.head(10))
        print(f"\nDataset summary:")
        print(df.describe())
        print(f"\nRecords per zip code:")
        print(df['zip_code'].value_counts())
        return df


# Create data directory if it doesn't exist
import os
os.makedirs('data', exist_ok=True)

# Execute data collection
zip_codes = ['07008', '10001', '19038', '95001']
collector = HouseDataCollector()

print("=" * 60)
print("HOUSE DATA COLLECTION")
print("=" * 60)
print(f"Target zip codes: {', '.join(zip_codes)}")

# Attempt web scraping
print("\n--- Attempting Web Scraping ---")
total_scraped = 0

for zip_code in zip_codes:
    try:
        count_before = len(collector.data)
        collector.scrape_redfin(zip_code, max_pages=2)
        count_after = len(collector.data)
        total_scraped += (count_after - count_before)
    except Exception as e:
        print(f"Error scraping {zip_code}: {e}")

print(f"\nTotal records scraped: {total_scraped}")

# If scraping didn't work well, generate synthetic data
if total_scraped < 100:
    print("\n⚠️  Web scraping yielded insufficient data.")
    print("Generating synthetic dataset for demonstration...")
    collector.data = []  # Reset
    collector.generate_synthetic_data(zip_codes)

# Save data to data directory
df = collector.save_to_csv('data/house_data.csv')

print("\nData collection complete!")
print(f"Dataset saved to: data/house_data.csv")