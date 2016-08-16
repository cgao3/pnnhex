#!/bin/bash
checkpoint=$1
 /home/ubuntu/tensorflow/bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=graph.pbtxt --input_checkpoint=$checkpoint --output_graph=const_graph.pb --output_node_names="x_input_node,conv4_layer/output_node"

