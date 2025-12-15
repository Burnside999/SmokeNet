# Dataset layout

All data is expected under the `dataset/` directory with two subfolders:

- `dataset/data/`: CSV files containing the raw sensor signals.
- `dataset/label/`: CSV files containing the labels for the corresponding signal files.

File names **must match exactly** between `data` and `label` (for example, `example.csv` must exist in both folders). The data loader asserts that the sets of stems are identical and will raise an error if any file is missing from either side.

Each CSV file is stored **without a header row**. Data files should contain the time-series values (arranged so that the two-dimensional tensor after loading represents channels × timesteps), and label files should contain one numeric value representing 0 for `no_fire` or 1 for `fire` at each timestep. (Note that the label file should have the **same number of rows** as the data file.)
