import pandas as pd
import gzip
import json

"""
Script taken from the source below to load amazon data into a pandas dataframe
Source: https://nijianmo.github.io/amazon/index.html
"""


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')
