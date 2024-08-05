/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <locale>
#include <codecvt>

#include "uima/api.hpp"
#include "llama.h"
#include "common.h"

using namespace uima;


#include "llamacpp-server-partial.inc"


class LlmAnnotator : public Annotator {
private:
  AnnotatorContext * pAnc;
  std::string promptStart;
  std::string promptEnd;
  gpt_params params;
  server_context ctx_server;
  std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};
  std::thread * server_queue;  
  

public:

  LlmAnnotator(void) {
    std::cout << "LlmAnnotator: Constructor" << std::endl;
  }

  ~LlmAnnotator(void) {
    std::cout << "LlmAnnotator: Destructor" << std::endl;
  }

  /** */
  TyErrorId initialize(AnnotatorContext & rclAnnotatorContext) {
    std::cout << "LlmAnnotator: initialize()" << std::endl;

    pAnc = &rclAnnotatorContext;
    
    icu::UnicodeString prompt;
    icu::UnicodeString variable;
    icu::UnicodeString model;
    std::string modelStr;
    
    if (!pAnc->isParameterDefined("Model") ||
        pAnc->extractValue("Model", model) != UIMA_ERR_NONE) {
      /* log the error condition */
      pAnc->getLogger().logError("Required configuration parameter \"Model\" not found in component descriptor");
      std::cout << "LlmAnnotator::initialize() - Error. See logfile." << std::endl;
      return UIMA_ERR_USER_ANNOTATOR_COULD_NOT_INIT;
    }
    model.toUTF8String(modelStr);

    if (!pAnc->isParameterDefined("Prompt") ||
        pAnc->extractValue("Prompt", prompt) != UIMA_ERR_NONE) {
      /* log the error condition */
      pAnc->getLogger().logError("Required configuration parameter \"Prompt\" not found in component descriptor");
      std::cout << "LlmAnnotator::initialize() - Error. See logfile." << std::endl;
      return UIMA_ERR_USER_ANNOTATOR_COULD_NOT_INIT;
    }

    if (!pAnc->isParameterDefined("Variable") ||
        pAnc->extractValue("Variable", variable) != UIMA_ERR_NONE) {
      /* log the error condition */
      pAnc->getLogger().logError("Required configuration parameter \"Variable\" not found in component descriptor");
      std::cout << "LlmAnnotator::initialize() - Error. See logfile." << std::endl;
      return UIMA_ERR_USER_ANNOTATOR_COULD_NOT_INIT;
    }

    int32_t pos = prompt.indexOf(variable);
    if (pos < 0) {
      /* log the error condition */
      pAnc->getLogger().logError("Variable not found on prompt");
      
      std::cout << "LlmAnnotator::initialize() - Error. Variable '" << variable << "' not found in prompt '" << prompt << "'" << std::endl;
      return UIMA_ERR_USER_ANNOTATOR_COULD_NOT_INIT;
    }

    icu::UnicodeString u_promptStart;
    icu::UnicodeString u_promptEnd;
    prompt.extract(0, pos, u_promptStart);
    auto endStart = pos + variable.length();
    prompt.extract(endStart, prompt.length() - endStart, u_promptEnd);
    u_promptStart.toUTF8String(promptStart);
    u_promptEnd.toUTF8String(promptEnd);

    /* log the configuration parameter setting */
    pAnc->getLogger().logMessage("Prompt = '" + prompt + "'");
    pAnc->getLogger().logMessage("Variable = '" + variable + "'");

    const char * argv[8] { "LlmAnnotator", "-m", modelStr.c_str(),
                           "-e", "--temp", "0", "--repeat-penalty", "1.0" };
    if (!gpt_params_parse(8, (char**)argv, params)) {
      pAnc->getLogger().logError("Model loading error");
      std::cout << "LlmAnnotator::initialize() - Error. See logfile." << std::endl;
      return UIMA_ERR_USER_ANNOTATOR_COULD_NOT_INIT;
    }

    // initialize server in ctx_server
    // (from llama.cpp / server.cpp / main
    if (!params.system_prompt.empty()) {
        ctx_server.system_prompt_set(params.system_prompt);
    }

    if (params.model_alias == "unknown") {
        params.model_alias = params.model;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    pAnc->getLogger().logMessage(std::string("build info: { build=") +
                                 std::to_string(LLAMA_BUILD_NUMBER)+
                                 std::string(", commit=") +
                                 std::string(LLAMA_COMMIT) +
                                 std::string(" }"));

    pAnc->getLogger().logMessage(std::string("system info: { n_threads=") +
                                 std::to_string(params.n_threads) +
                                 std::string(", n_threads_batch=")+
                                 std::to_string(params.n_threads_batch)+
                                 std::string(", total_threads=")+
                                 std::to_string(std::thread::hardware_concurrency())+
                                 std::string(", system_info=")+
                                 std::string(llama_print_system_info())+
                                 std::string(" }"));
    
    // Necessary similarity of prompt for slot selection
    ctx_server.slot_prompt_similarity = params.slot_prompt_similarity;

    // load the model
    if (!ctx_server.load_model(params)) {
        state.store(SERVER_STATE_ERROR);
        pAnc->getLogger().logError("Model loading error");
        std::cout << "LlmAnnotator::initialize() - Error. See logfile." << std::endl;
        return UIMA_ERR_USER_ANNOTATOR_COULD_NOT_INIT;
    } else {
        ctx_server.init();
        state.store(SERVER_STATE_READY);
    }

    pAnc->getLogger().logMessage("model loaded");
    const auto model_meta = ctx_server.model_meta();

    // if a custom chat template is not supplied, we will use the one that comes with the model (if any)
    if (params.chat_template.empty()) {
        if (!ctx_server.validate_model_chat_template()) {
          pAnc->getLogger().logWarning("The chat template that comes with this model is not yet supported, falling back to chatml. This may cause the model to output suboptimal responses");
          params.chat_template = "chatml";
        }
    }

    // print sample chat example to make it clear which template is used
    {
    pAnc->getLogger().logMessage(std::string("chat template:: { chat_example=") +
                                 llama_chat_format_example(ctx_server.model, params.chat_template) +
                                 std::string(", built_in=")+
                                 std::to_string(params.chat_template.empty())+
                                 std::string(" }"));
    }

    ctx_server.queue_tasks.on_new_task(std::bind(
        &server_context::process_single_task, &ctx_server, std::placeholders::_1));
    ctx_server.queue_tasks.on_finish_multitask(std::bind(
        &server_context::on_finish_multitask, &ctx_server, std::placeholders::_1));
    ctx_server.queue_tasks.on_update_slots(std::bind(
        &server_context::update_slots, &ctx_server));
    ctx_server.queue_results.on_multitask_update(std::bind(
        &server_queue::update_multitask,
        &ctx_server.queue_tasks,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3
    ));
   

    server_queue = new std::thread([&](){ ctx_server.queue_tasks.start_loop(); });

    LOG_TEE("sampling: \n%s\n", llama_sampling_print(params.sparams).c_str());
    LOG_TEE("sampling order: \n%s\n", llama_sampling_order_print(params.sparams).c_str());
    LOG_TEE("\n");
    LOG_TEE("%s\n", gpt_params_get_system_info(params).c_str());
    LOG_TEE("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", llama_n_ctx(ctx_server.ctx), params.n_batch, params.n_predict, params.n_keep);
    
    std::cout << "LlmAnnotator::initialize() .. promptStart: "
         << promptStart << " .. promptEnd: " << promptEnd << std::endl;

    return (TyErrorId)UIMA_ERR_NONE;
  }

  /** */
  TyErrorId typeSystemInit(TypeSystem const & crTypeSystem) {
    std::cout << "LlmAnnotator:: typeSystemInit()" << std::endl;
    return (TyErrorId)UIMA_ERR_NONE;
  }

  /** */
  TyErrorId destroy() {
    std::cout << "LlmAnnotator: destroy()" << std::endl;
    ctx_server.queue_tasks.terminate();
    server_queue->join();
    llama_backend_free();
    return (TyErrorId)UIMA_ERR_NONE;
  }

  /** */
  TyErrorId process(CAS & rCas, ResultSpecification const & crResultSpecification) {
    std::cout << "LlmAnnotator::process() begins" << std::endl;

    /** get the CAS view of the sofa */
    CAS * tcas = rCas.getView(CAS::NAME_DEFAULT_SOFA);

    /* This is a shallow pointer object containing a reference to document text*/
    UnicodeStringRef doc = tcas->getDocumentText();

    std::string fullPrompt;
    fullPrompt += promptStart;
    fullPrompt += doc.asUTF8();
    fullPrompt += promptEnd;

    std::cout << "LlmAnnotator::process() UTF-8 full prompt length " << fullPrompt.length() << " characters " << std::endl;
    
    // call the LLM
    json prompt { { "prompt", fullPrompt } };
    std::cout << prompt << std::endl;
    const int id_task = ctx_server.queue_tasks.get_new_id();    

    ctx_server.queue_results.add_waiting_task_id(id_task);
    ctx_server.request_completion(id_task, -1, prompt, false, false);

    server_task_result result = ctx_server.queue_results.recv(id_task);
    std::cout << "LlmAnnotator::process() got LLM result " << std::endl;
    //TODO result.stop could be set as an annotation
    std::string llmOutput;
    if (!result.error) {
      llmOutput = result.data["content"];
    }else{
      llmOutput = result.data.dump(-1, ' ', false, json::error_handler_t::replace);
    }
    ctx_server.queue_results.remove_waiting_task_id(id_task);

    icu::UnicodeString us_llmOutput = icu::UnicodeString::fromUTF8(llmOutput);

    // Create the output LLM text Sofa and open CAS view
    CAS * llmTcas = rCas.createView("LlmOutput");

    llmTcas->setDocumentText( us_llmOutput.getBuffer(), us_llmOutput.length(), true );

    std::cout << "LlmAnnotator::process() ends" << std::endl;
    return (TyErrorId)UIMA_ERR_NONE;
  }
};

// This macro exports an entry point that is used to create the annotator.

MAKE_AE(LlmAnnotator);
