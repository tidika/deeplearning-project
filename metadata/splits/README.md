# Split Files

Store committed image split files in this directory.

Recommended files:

- `train.txt`
- `val.txt`
- `test.txt`

Each file should contain one image filename per line, for example:

```text
1000092795.jpg
10002456.jpg
1000268201.jpg
```

Keep splits image-level, not caption-level, to avoid train/test leakage.
