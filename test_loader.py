from src.data_loader import load_raw

df = load_raw()

print(df.head())
print(df.shape)
print(df.dtypes)