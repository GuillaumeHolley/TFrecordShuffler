# Local in-memory TensorFlow records shuffling with or without external storage
# Guillaume Holley, deCODE Genetics, 2021.

import os
import sys
import glob
import random

import argparse
import hashlib
import logging

import zlib
import pickle

import tensorflow.compat.v1 as tf

def parse_cmdline(argv):

	parser = argparse.ArgumentParser()

	parser.add_argument('--input_pattern_list', help='Comma-separated list of TFRecord filename patterns.')
	parser.add_argument('--output_pattern_prefix', help='Filename pattern for the output TFRecords.')
	parser.add_argument('--output_dataset_config_pbtxt', help='Optional.  If set, print out a human-readable version of DeepVariantDatasetConfig.')
	parser.add_argument('--output_dataset_name', help='Optional unless --output_dataset_config_pbtxt is set.')
	parser.add_argument('--direct_num_workers', help='Optional. If set, output will be split in that many worker files.', type=int, default=1)
	parser.add_argument('--step', help='Optional. Configure how many TF records can be loaded in memory at once. -1 is all of them.', type=int, default=-1)

	known_args, pipeline_args = parser.parse_known_args(argv)

	return known_args, pipeline_args

def shuffle_tfrecords(dataset_name, input_filename_pattern_list, input_pattern_list, output_pattern_prefix, output_config_filename, direct_num_workers, step):

	def sha1(input_bytes):

		m = hashlib.sha1()
		m.update(input_bytes)

		return m.digest()

	def label_example(input_bytes):

		example = tf.train.Example.FromString(input_bytes)
		label = example.features.feature['label'].int64_list.value[0]

		return label

	i = 0
	num_records = 0
	records = []
	lfn = []
	lrec = []
	compression = tf.python_io.TFRecordCompressionType.GZIP

	# Make sure we process the input files always in the same order
	for idx, filepattern in enumerate(input_filename_pattern_list): lfn.extend(glob.glob(filepattern))
		
	if (len(lfn) == 0): sys.exit("No file found matching input pattern list. Shuffling aborted.")

	lfn.sort()

	# Hash records and compute their labels
	print("-> Reading input")

	for fn in lfn:

		print("--> Processing " + fn)

		for rec in tf.io.tf_record_iterator(fn, tf.python_io.TFRecordOptions(compression)):

			h = sha1(rec)
			label = label_example(rec)

			records.append((h, i, label))

			if (step == -1): lrec.append(zlib.compress(pickle.dumps(rec)))

			i += 1

	num_records = i

	print("-> Shuffling TF records")
	records.sort(key=lambda x: x[0]) # Sort record by their sha1 hash

	# Count how many labels there are (total and per class)
	print("-> Counting labels")

	label_counts = {}
	num_examples = 0
	num_examples_by_labels = ""

	for rec in records:

		if (rec[2] not in label_counts): label_counts[rec[2]] = 0

		label_counts[rec[2]] += 1

	for label, count in label_counts.items(): 

		num_examples_by_labels += "# class" + str(label) + ": " + str(count) + '\n'
		num_examples += count

	del label_counts

	worker_id = 0
	num_record_worker = 0
	num_record_total = 0
	num_records_per_thread = int(num_records / direct_num_workers) + 1
	num_records_print = int(num_records / 20)

	output_examples = output_pattern_prefix + "-" + "{:05d}".format(worker_id) + "-of-" + "{:05d}".format(direct_num_workers) + ".tfrecord.gz"
	writer = tf.python_io.TFRecordWriter(output_examples, options=tf.python_io.TFRecordOptions(compression))

	if (step == -1): # User explicitly instructed to load ALL records in memory

		# Write shuffled records to disk
		print("-> Writing shuffled TF records to output")

		for rec in records:

			zdRec = pickle.loads(zlib.decompress(lrec[rec[1]]))

			writer.write(zdRec)

			num_record_worker += 1
			num_record_total += 1

			if (num_record_total % num_records_print == 0): print("--> Written " + str(num_record_total) + " / " + str(num_records) + " records to final location")

			if (num_record_worker >= num_records_per_thread):

				worker_id += 1
				num_record_worker = 0

				writer.close()

				output_examples = output_pattern_prefix + "-" + "{:05d}".format(worker_id) + "-of-" + "{:05d}".format(direct_num_workers) + ".tfrecord.gz"
				writer = tf.python_io.TFRecordWriter(output_examples, options=tf.python_io.TFRecordOptions(compression))		
	
	else: # User explicitely gave the number of TF records that could be loaded in memory at once

		# Write shuffled records
		print("-> Reordering positions of shuffled TF records")

		i = 0
		pos_shuff = {}

		for rec in records:

			pos_shuff[rec[1]] = i
			i += 1

		del records # Not required anymore

		# Write shuffled records to disk
		print("-> Writing shuffled TF records to output")

		if (step < num_records): # Cannot hold all the TF records in memory at once, need some tmp storage

			num_open_files = int(num_records / step) + 1
			fo = [] # Keep track of opened files
			bsz = [] # Keep track of the size of byte objects written to disk
			pos_shuff_bucket = []

			for i in range(0, num_open_files):

				fon = output_pattern_prefix + "_tmp_" + str(i) + ".cpickle"

				fo.append(open(fon, "wb"))
				pos_shuff_bucket.append([])
				bsz.append([])

			i = 0
			lrec = []
			lrec_pos = []

			for fn in lfn:

				for rec in tf.io.tf_record_iterator(fn, tf.python_io.TFRecordOptions(compression)):

					j = pos_shuff[i]
					zcpRec = zlib.compress(pickle.dumps(rec))

					lrec.append(zcpRec)
					lrec_pos.append((j, i % step))

					i += 1
					num_record_total += 1

					if (i % step == 0):

						lrec_pos.sort()

						for new_pos, curr_pos in lrec_pos:
					
							bucket = int(new_pos / step)
							sz = fo[bucket].write(lrec[curr_pos])

							pos_shuff_bucket[bucket].append(new_pos)
							bsz[bucket].append(sz)

						lrec = []
						lrec_pos = []

					if (num_record_total % num_records_print == 0): print("--> Written " + str(num_record_total) + " / " + str(num_records) + " records to tmp location (1/2)")

			if (len(lrec) != 0):

				lrec_pos.sort()

				for new_pos, curr_pos in lrec_pos:
			
					bucket = int(new_pos / step)
					sz = fo[bucket].write(lrec[curr_pos])

					pos_shuff_bucket[bucket].append(new_pos)
					bsz[bucket].append(sz)

			for i in range(0, num_open_files): fo[i].close()

			del pos_shuff

			num_record_total = 0

			for i in range(0, num_open_files): 	

				fon = output_pattern_prefix + "_tmp_" + str(i) + ".cpickle"
				fo = open(fon, "rb")

				lrec = []
				lrec_pos = []
				j = 0

				for sz in bsz[i]:

					zcRec = bytes(fo.read(sz)) # Read sz bytes from input file, convert to bytes object -> compressed pickled TF record

					lrec.append(zcRec)
					lrec_pos.append((pos_shuff_bucket[i][j], j))

					j += 1

				lrec_pos.sort()

				for new_pos, curr_pos in lrec_pos:

					zdRec = pickle.loads(zlib.decompress(lrec[curr_pos]))

					writer.write(zdRec)

					num_record_worker += 1
					num_record_total += 1

					if (num_record_total % num_records_print == 0): print("--> Written " + str(num_record_total) + " / " + str(num_records) + " records to final location (2/2)")

					if (num_record_worker >= num_records_per_thread):

						worker_id += 1
						num_record_worker = 0

						writer.close()

						output_examples = output_pattern_prefix + "-" + "{:05d}".format(worker_id) + "-of-" + "{:05d}".format(direct_num_workers) + ".tfrecord.gz"
						writer = tf.python_io.TFRecordWriter(output_examples, options=tf.python_io.TFRecordOptions(compression))

			print("-> Clean tmp files")

			for i in range(0, num_open_files):

				fon = output_pattern_prefix + "_tmp_" + str(i) + ".cpickle"
				os.remove(fon)

		else:

			i = 0
			lrec = []
			lrec_pos = []

			for fn in lfn:

				for rec in tf.io.tf_record_iterator(fn, tf.python_io.TFRecordOptions(compression)):

					j = pos_shuff[i]
					zcRec = zlib.compress(pickle.dumps(rec))

					lrec.append(zcRec)
					lrec_pos.append((j, i))

					i += 1
					num_record_total += 1

					if (num_record_total % num_records_print == 0): print("--> Loaded " + str(num_record_total) + " / " + str(num_records) + " records in memory (1/2)")

			del pos_shuff

			lrec_pos.sort()

			num_record_total = 0

			for new_pos, curr_pos in lrec_pos:

				zdRec = pickle.loads(zlib.decompress(lrec[curr_pos]))

				writer.write(zdRec)

				num_record_worker += 1
				num_record_total += 1

				if (num_record_total % num_records_print == 0): print("--> Written " + str(num_record_total) + " / " + str(num_records) + " records to final location (2/2)")

				if (num_record_worker >= num_records_per_thread):

					worker_id += 1
					num_record_worker = 0

					writer.close()

					output_examples = output_pattern_prefix + "-" + "{:05d}".format(worker_id) + "-of-" + "{:05d}".format(direct_num_workers) + ".tfrecord.gz"
					writer = tf.python_io.TFRecordWriter(output_examples, options=tf.python_io.TFRecordOptions(compression))

	print("-> Create DeepVariant summary file")

	# Get absolute path prefix from output_pattern_prefix and input_pattern_list files
	output_examples = output_pattern_prefix + "-00000-of-" + "{:05d}".format(direct_num_workers) + ".tfrecord.gz"
	abspath_output_examples = os.path.abspath(output_examples)
	abspath_out_prefix_sz = len(abspath_output_examples) - len(output_examples)

	abspath_input_examples = os.path.abspath(lfn[0])
	abspath_in_prefix_sz = len(abspath_input_examples) - len(input_pattern_list)
	
	if (abspath_out_prefix_sz != 0): output_pattern_prefix = abspath_output_examples[0:abspath_out_prefix_sz] + output_pattern_prefix
	if (abspath_in_prefix_sz != 0): input_pattern_list = abspath_input_examples[0:abspath_in_prefix_sz] + input_pattern_list

	# Create summary file
	fo = open(output_config_filename, "w")

	fo.write("# Generated by shuffle_tfrecords_lowmem.py\n") # Write header
	fo.write("\n")
	fo.write("name: \"" + dataset_name + "\"\n") # Write dataset name
	fo.write("tfrecord_path: \"" + output_pattern_prefix + "-?????-of-?????.tfrecord.gz\"\n") # Write output pattern prefix
	fo.write("num_examples: " + str(num_examples) + "\n") # Write total number of examples
	fo.write("#\n")
	fo.write("# --input_pattern_list=" + input_pattern_list + "\n") # Write input file pattern
	fo.write("# --output_pattern_prefix=" + output_pattern_prefix + "\n") # Write output file pattern
	fo.write("#\n")
	fo.write(num_examples_by_labels) # Write number of examples by label

	fo.close()

if __name__ == '__main__':

	logging.getLogger().setLevel(logging.INFO)

	known_args, pipeline_args = parse_cmdline(sys.argv)

	input_examples = shuffle_tfrecords(
		known_args.output_dataset_name,
		known_args.input_pattern_list.split(','),
		known_args.input_pattern_list,
		known_args.output_pattern_prefix,
		known_args.output_dataset_config_pbtxt,
		known_args.direct_num_workers,
		known_args.step
	)
