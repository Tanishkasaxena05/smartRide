import sys
sys.path.insert(0, '.')
from src.data_loader import load_data, clean_data, add_time_period
from src.dbscan_clustering import run_dbscan

df = load_data()
df = clean_data(df)
df = add_time_period(df)

r = run_dbscan(df, eps_meters=200, min_samples=900)
noise = (r['cluster'] == -1).sum()
print('NOISE COUNT:', noise)