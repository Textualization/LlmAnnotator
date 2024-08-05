#######################################
# UNIX Makefile for a UIMACPP annotator
#######################################

# ---------------------------------------------------------------------------
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------

# name of the annotator to be created
TARGET_FILE=LlmAnnotator

# list of user's object files to be linked when building the annotator
OBJS=LlmAnnotator.o \
  llama.cpp/src/llama-grammar.o llama.cpp/src/llama-vocab.o llama.cpp/src/unicode-data.o llama.cpp/src/unicode.o llama.cpp/src/llama-sampling.o llama.cpp/src/llama.o  \
  llama.cpp/ggml/src/ggml-aarch64.o  llama.cpp/ggml/src/ggml-backend.o  llama.cpp/ggml/src/ggml.o llama.cpp/ggml/src/ggml-alloc.o    llama.cpp/ggml/src/ggml-quants.o llama.cpp/ggml/src/llamafile/sgemm.o \
  llama.cpp/common/common.o llama.cpp/common/build-info.o llama.cpp/common/sampling.o llama.cpp/common/grammar-parser.o  llama.cpp/common/json-schema-to-grammar.o   llama.cpp/common/ngram-cache.o llama.cpp/common/console.o

#Use this var to pass additional user-defined parameters to the compiler
USER_CFLAGS=-Illama.cpp/common -Illama.cpp/spm-headers -Illama.cpp/examples/server -Wno-unused-function

#Use this var to pass additional user-defined parameters to the linker
USER_LINKFLAGS=-fopenmp

# Set DEBUG=1 for a debug build (if not 1 a ship build will result)
DEBUG=1

# Set DLL_BUILD=1 to build an annotator (shared library)
#    if not 1 an executable binary will be built
DLL_BUILD=1

# include file with generic compiler instructions
include $(UIMACPP_HOME)/lib/base.mak
