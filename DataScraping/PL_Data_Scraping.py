import re
import time
import pandas as pd
from bs4 import BeautifulSoup, Comment

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

BASE_URL = "https://fbref.com"
SEASON_URL = f"{BASE_URL}/en/comps/9/Premier-League-Stats"
OUTPUT_CSV = "stats.csv"


def make_driver(headless=False):
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def get_soup(driver, url, wait_sec=5):
    driver.get(url)
    time.sleep(wait_sec)
    return BeautifulSoup(driver.page_source, "lxml"), driver.page_source


def extract_team_urls(soup):
    urls = set()

    # Normal anchors
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if "/en/squads/" in h and h.endswith("-Stats"):
            urls.add(BASE_URL + h if h.startswith("/") else h)

    # Anchors inside HTML comments (FBref pattern)
    comments = soup.find_all(string=lambda t: isinstance(t, Comment))
    for c in comments:
        if "/en/squads/" not in c:
            continue
        cs = BeautifulSoup(c, "lxml")
        for a in cs.find_all("a", href=True):
            h = a["href"]
            if "/en/squads/" in h and h.endswith("-Stats"):
                urls.add(BASE_URL + h if h.startswith("/") else h)

    return sorted(urls)


def find_team_stats_table(soup):
    # direct tables
    tables = soup.find_all("table", class_="stats_table")
    if tables:
        return tables[0]

    # tables inside comments
    comments = soup.find_all(string=lambda t: isinstance(t, Comment))
    for c in comments:
        if "stats_table" in c:
            cs = BeautifulSoup(c, "lxml")
            t = cs.find("table", class_="stats_table")
            if t:
                return t
    return None


def main():
    driver = make_driver(headless=False)  # run visible browser first
    all_teams = []

    try:
        season_soup, season_html = get_soup(driver, SEASON_URL, wait_sec=6)
        team_urls = extract_team_urls(season_soup)

        if not team_urls:
            with open("debug_season_page.html", "w", encoding="utf-8") as f:
                f.write(season_html)
            raise RuntimeError(
                "No team URLs found. Saved page to debug_season_page.html "
                "(likely challenge/blocked page)."
            )

        print(f"Found {len(team_urls)} team URLs")

        for team_url in team_urls:
            team_name = team_url.split("/")[-1].replace("-Stats", "")
            print(f"Scraping {team_name}...")

            soup, _ = get_soup(driver, team_url, wait_sec=4)
            table = find_team_stats_table(soup)
            if table is None:
                print(f"  Skipped {team_name}: no stats table")
                continue

            df = pd.read_html(str(table))[0]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel()

            df["Team"] = team_name
            all_teams.append(df)
            time.sleep(1.5)

        if not all_teams:
            raise RuntimeError("No team data scraped.")

        final_df = pd.concat(all_teams, ignore_index=True)
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved {OUTPUT_CSV} with {len(final_df)} rows.")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
