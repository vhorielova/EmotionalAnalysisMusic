import pandas as pd
from sklearn.model_selection import train_test_split
from config import VAL_SIZE, RANDOM_STATE

def split_ids(labels, train_ids, test_ids):
    train_ids = train_ids.tolist()
    test_ids = test_ids.tolist()

    y_train_ids = pd.Series([labels[s] for s in train_ids])

    train_ids, val_ids = train_test_split(train_ids, test_size=VAL_SIZE, random_state=RANDOM_STATE,
                                          stratify=y_train_ids)

    print("Test size: ", len(test_ids))
    print("Train size: ", len(train_ids))
    print("Validation size: ", len(val_ids))

    print(pd.Series([labels[s] for s in test_ids]).value_counts(normalize=True))
    print(y_train_ids.value_counts(normalize=True))
    print(pd.Series([labels[s] for s in val_ids]).value_counts(normalize=True))

    print(set(train_ids) & set(val_ids), set(train_ids) & set(test_ids), set(val_ids) & set(test_ids))

    return train_ids, val_ids, test_ids