import sys
sys.path.insert(0, '.')
from src.data_loader import load_raw_tables, build_master_df, add_parsed_lap_times

print("Loading tables...")
tables = load_raw_tables()

print()
print("Table shapes:")
for name, df in tables.items():
    print(f"  {name:30s} {str(df.shape):15s}")

print()
print("Building master dataframe...")
df = build_master_df(tables)
df = add_parsed_lap_times(df)
print(f"  Master DF shape: {df.shape}")
print(f"  Years covered:   {int(df['year'].min())} - {int(df['year'].max())}")
print(f"  Total races:     {df['raceId'].nunique()}")
print(f"  Total drivers:   {df['driverId'].nunique()}")
print()
print("All good - data loaded successfully!")
