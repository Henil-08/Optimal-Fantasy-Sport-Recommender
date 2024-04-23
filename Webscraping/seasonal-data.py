import pandas as pd  # file operations
from bs4 import BeautifulSoup  #Scrapping tool
from urllib.request import urlopen as ureq # For requesting data from link
import numpy as np
import re
import csv

import requests


for i in range(2019, 2024):
    for inning in ['bat, bowl']:
        try:
            url = f"http://www.howstat.com/cricket/Statistics/IPL/SeriesAnalysis.asp?s={i}#{inning}"
            response = requests.get(url)
            response.raise_for_status()  # Check for successful response

            # Parse the HTML content
            soup = BeautifulSoup(response.content, "html.parser")

            # Find the table
            # table = soup.find("table", {"class": "TableLined"})
            table = soup.find("div", {"id": inning})
            table = table.find("table", {"class": "TableLined"})

        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: {e}")

        except requests.exceptions.HTTPError as e:
            print(f"HTTP error: {e}")

        for x in table:
            rows = table.find_all('tr') #find all tr tag(rows)
        for tr in rows:
            data=[]
            cols = tr.find_all('td') #find all td tags(columns)
            for td in cols:
                data.append(td.text.strip())
                
        with open(f"IPL{i}{inning}.csv", 'a', newline='') as csvfile:
            f = csv.writer(csvfile)
            for x in table:
                rows = table.find_all('tr') #find all tr tag(rows)
            for tr in rows:
                data=[]
                cols = tr.find_all('td') #find all td tags(co)
                for td in cols:
                    data.append(td.text.strip())
                f.writerow(data)
