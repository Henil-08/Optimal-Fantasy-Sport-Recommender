import pandas as pd  # file operations
from bs4 import BeautifulSoup  #Scrapping tool
from urllib.request import urlopen as ureq # For requesting data from link
import numpy as np
import re
import csv

import requests

teams = ['CSK', 'MI', 'RCB', 'SRH', 'DC', 'KKR', 'RR', 'PK', 'GT', 'LSG']
data = []

for i in range(2019, 2024):
    for team in teams:
        try:
            url = f"http://www.howstat.com/cricket/Statistics/IPL/PlayerList.asp?s={i}&t={teams}"
            response = requests.get(url)
            response.raise_for_status()  # Check for successful response

            # Parse the HTML content
            soup = BeautifulSoup(response.content, "html.parser")

            # Find the table
            table = soup.find("table", {"class": "TableLined"})

        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: {e}")

        except requests.exceptions.HTTPError as e:
            print(f"HTTP error: {e}")

                
        with open(f"IPL_Squads.csv", 'a', newline='') as csvfile:
            f = csv.writer(csvfile)
            for x in table:
                rows = table.find_all('tr') #find all tr tag(rows)
            for tr in rows:
                cols = tr.find_all('LinkTable') #find all td tags(co)
                for td in cols:
                    data.append(td.text.strip())
            f.writerow(i, teams, data)
