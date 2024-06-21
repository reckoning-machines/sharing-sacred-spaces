# newsapi.ai - event registry python package
from eventregistry import *
# import pandas as pd
import json

er = EventRegistry(apiKey = "")

q = QueryArticlesIter(
    keywords = "hate crime",
    keywordsLoc = "body,title",
    lang = "eng",
    sourceLocationUri = er.getLocationUri("USA"),
    dateStart = "2024-05-05"
    # isDuplicateFilter = "skipDuplicates"
)

filename = "articles_metadata.txt"

try:
    with open(filename, "w") as file:
        for article in q.execQuery(er, 
                                sortBy = "date", 
                                returnInfo = ReturnInfo(articleInfo = ArticleInfoFlags(body=False, location = True, image=False),
                                                        locationInfo=LocationInfoFlags(
                                    geoNamesId=True,
                                    population=True,
                                    geoLocation=True,
                                    countryDetails=True,
                                    placeCountry = False
                                ))):
            # Write JSON response to the file
            json.dump(article, file, indent=4)
            file.write("\n\n")
    
    print(f"Article metadata saved to {filename}")

except Exception as e:
    print(f"An error occurred: {e}")


# pygooglenews - python wrapper of google news rss feed
from pygooglenews import GoogleNews
import json
import time

gn = GoogleNews('en','US')
hate = gn.search('hate crime', from_ = '2014-01-01', to_ = '2024-06-05')
entries = hate["entries"]
count = 0
for entry in entries:
  count = count + 1
  print(
    str(count) + ". " + entry["title"] + entry["published"]
  )
  time.sleep(0.1)