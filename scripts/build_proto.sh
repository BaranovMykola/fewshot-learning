#!/bin/bash

protoc ./src/dataset/proto/few_shot_sample/few_shot_sample.proto --python_out=. \
    --descriptor_set_out=./src/dataset/proto/few_shot_sample/few_shot_sample.desc \
    --include_imports

