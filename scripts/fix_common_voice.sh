#!/bin/sh

# since lhotse is taking wrong indexes of columns for common voice it's easier to just fix the dataset metadata
# already created an issue on that: https://github.com/lhotse-speech/lhotse/issues/1325

TRAIN_TSV_PATH=data/cv-corpus-17.0-2024-03-15-pl/cv-corpus-17.0-2024-03-15/pl/train.tsv
TEST_TSV_PATH=data/cv-corpus-17.0-2024-03-15-pl/cv-corpus-17.0-2024-03-15/pl/test.tsv
TRAIN_TSV_PATH_CPY=data/cv-corpus-17.0-2024-03-15-pl/cv-corpus-17.0-2024-03-15/pl/train_copy.tsv
TEST_TSV_PATH_CPY=data/cv-corpus-17.0-2024-03-15-pl/cv-corpus-17.0-2024-03-15/pl/test_copy.tsv

cut -f 3 --complement $TRAIN_TSV_PATH_CPY > temp_train.tsv
cut -f 3 --complement $TEST_TSV_PATH_CPY > temp_test.tsv
mv temp_train.tsv $TRAIN_TSV_PATH
mv temp_test.tsv $TEST_TSV_PATH