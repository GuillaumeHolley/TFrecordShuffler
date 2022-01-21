# TFrecordShuffler

The script shuffles TensorFlow records locally and in-memory using as little RAM as possible, with or without external storage. It is intended to be used as a drop-in replacement for `shuffle_tfrecords_beam.py` in the DeepVariant training instructions when using the cloud is not an option. The key advantage of this script is that it very memory-efficient but it is not parallelized so it takes much longer to shuffle the records.

## Requirements

Python 3 is required and TensorFlow.
```
pip3 install tensorflow
```
Other modules are required (`zlib, pickle, glob, hashlib, logging`) but should already be installed by default.

## Usage

```
python3 shuffle_tfrecords_lowmem.pu \
--input_pattern_list="deepvariant_training/training_set.with_label.tfrecord-?????-of-00024.gz" \
--output_pattern_prefix="deepvariant_training/training_set.with_label.shuffled" \
--output_dataset_config="training_set.pbtxt" \
--output_dataset_name="HG002" \
--direct_num_workers=24 \
--step=-1
```

### Output

The shuffled TF record files will have the prefix `deepvariant_training/training_set.with_label.shuffled`. There should be `direct_num_workers`. Furthermore, a summary file for DeepVariant training will be generated in `training_set.pbtxt`.

### Memory usage

* **Without external storage**

By default, all TF records are shuffled in memory at once (`--step=-1`). If the files matching the input pattern list `deepvariant_training/training_set.with_label.tfrecord-?????-of-00024.gz` take a total of X GB on disk, you will need about 1.2 * X GB of RAM.

* **With external storage**

You can use `--step=Y` to specify that only Y TF records can be loaded in memory at once. Unfortunately it is impossible to say beforehand how much memory you need for Y TF records so it is a test-and-try parameter for now.

### Improvements

There is a lot of space for improvement, especially in terms of parallelization so so I welcome any PR.
