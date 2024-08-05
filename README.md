# UIMACPP LLM Annotator

Building instructions: 

* In a UIMACPP build docker image
* `cd llama.cpp; make -j`
* in this folder `make`

Test with 

Download gemma-2-2b-it.q2_k.gguf from Huggingface, then:

`export LD_LIBRARY_PATH=$PWD; /usr/local/uima/ae -l 2 LlmAnnotator.xml sample.txt out`


