# This is a sample Python script.

# Press ⌃F5 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import json
import pandas as pd

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press F9 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with open("data/bronze/news/NVDA.json") as f:
        articles = json.load(f)
    print("total:", len(articles))
    print(json.dumps(articles[0], indent=2))  # one full article
    # check how many have non-empty summaries (the text we embed)
    with_summary = sum(1 for a in articles if a.get("summary", "").strip())
    print(f"articles with non-empty summary: {with_summary}/{len(articles)}")
