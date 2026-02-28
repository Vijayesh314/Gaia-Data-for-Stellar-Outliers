from astropy.io import votable

# Load the VOTable
votable_data = votable.parse("dataset.vot")

# Convert to table
table = votable_data.get_first_table().to_table()

# Convert to pandas DataFrame
df = table.to_pandas()

print(df.shape)
print(df.columns)
df.head()

df.to_csv("gaia_clean_sample.csv", index=False)