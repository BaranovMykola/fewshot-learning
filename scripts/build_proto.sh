#!/bin/bash

protoc ./src/dataset/proto/fss_dataset.proto --python_out=. --descriptor_set_out=./src/dataset/proto/fss_dataset.desc \
     --include_imports

