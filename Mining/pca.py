# %%
import pandas as pd

from sklearn.decomposition import PCA

# %%
data_df = pd.read_csv("../final_processed_data.csv")
data_df

# %%
X = data_df.drop(["track_id", "bit_rate", "genre_top", "file_path"], axis=1)
y = data_df[["genre_top"]]

# %%
X

# %%
y

# %%
pca = PCA(n_components=5)
pca.fit(X)
X_reduced = pca.transform(X)
X_reduced

# %%
data_reduced = pd.concat([y, pd.DataFrame(X_reduced)], axis=1)
data_reduced

# %%
data_reduced.to_csv("./data_PCA_5comp.csv", index=False)

# %%
