import pandas as pd  # file operations
from bs4 import BeautifulSoup  #Scrapping tool
from urllib.request import urlopen as ureq # For requesting data from link
import numpy as np
import re
import csv

import requests

Matchcode_start = 0705
Matchcode_end = 1033
data = []

for i in range(Matchcode_start, Matchcode_end):
    try:
        url = f"https://sportstar.thehindu.com/cricket/ipl/ipl-news/article68050938.ece?MatchCode={i}"
        response = requests.get(url)
        response.raise_for_status()  # Check for successful response

        # Parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")

        # Find the table
        table = soup.find("div", {"class": "fact-box"})

    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")

    for x in table:
        rows = table.find('fact-element') #find all tr tag(rows)
    

    with open(f"IPL_DreamTeam.csv", 'a', newline='') as csvfile:

        f = csv.writer(csvfile)
        for x in table:
            rows = table.find('fact-element') #find all tr tag(rows)
            data.append(rows)
        f.writerow(i, data)
