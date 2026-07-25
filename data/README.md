# Data directory

The repository does not redistribute AortaSeg24, BTCV, or any patient data.
Obtain each dataset from its original source and follow its data-use terms.

For the simplest directory-mode run, place one or more compressed 3D NIfTI
volumes directly in `data/input/`:

```text
data/
├── input/
│   ├── case_0001.nii.gz
│   ├── case_0002.nii.gz
│   └── case_0003.nii.gz
└── output/
```

Dataset-specific names are also accepted, for example:

```text
data/input/AortaSeg24_0001.nii.gz
data/input/AortaSeg24_0001_0000.nii.gz
data/input/img0001.nii.gz
```

The program does not impose a case-name pattern. A file must end in `.nii.gz`
for directory mode, and every output keeps the exact input filename.

For JSON mode, edit `configs/dataset_example.json`. Each item must contain an
`image` path. A relative path is resolved from the directory containing the
JSON file.
