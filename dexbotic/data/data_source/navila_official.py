from dexbotic.data.data_source.register import register_dataset

NAVILA_DATASET = {
    "R2R": {
        "data_path_prefix": "./data/navila/R2R/images",
        "annotations": "./data/navila/R2R",
        "frequency": 1,
    },
}

meta_data = {}

register_dataset(NAVILA_DATASET, meta_data=meta_data, prefix="navila")
