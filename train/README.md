# Fine-tuning

## Content
- [Fine-tuning](#fine-tuning)
  - [Content](#content)
  - [Training Data Creation](#training-data-creation)
    - [Format](#format)
    - [Generate New Training Data](#generate-new-training-data)
      - [Multi-hop Questions asking about an attribute of 2 entities in a question](#multi-hop-questions-asking-about-an-attribute-of-2-entities-in-a-question)
      - [Multi-hop Questions asking about 2 attributes of an entity in a question](#multi-hop-questions-asking-about-2-attributes-of-an-entity-in-a-question)
  - [Installation](#installation)
  - [Training script](#training-script)
    - [Single GPU](#single-gpu)
    - [Multiple GPU](#multiple-gpu)
  - [Evaluation](#evaluation)
  - [Training Data](#training-data)
    - [Format](#format-1)
    - [Single questions:](#single-questions)
    - [Multi-hop questions](#multi-hop-questions)
      - [Musique](#musique)
      - [Generate training data](#generate-training-data)
      - [Script to generate Multi-hop training data](#script-to-generate-multi-hop-training-data)

## Training Data Creation 
Each multi-hop question can be handled by decomposing it into single questions. This datasets contains multi-hop questions and their decomposed questions. We also add single questions to this dataset to make sure that the trained model is able to handle all kinds of questions.

### Format 
Each data point is a Json with fields:
+ **question**: the question, can be single question or multi-hop question
+ **multihop**: True/False whether the question is multihop or not 
+ **sub_questions**: List of decomposed single questions from question. If the question is single question, ```len(sub_questions) == 1```
    + **question**: single question decomposed from original multi-hop question
    + **paragraph**: the retrieval context for the single question
    + **long_answer**: the answer to the single question, the format is: xxx\nAnswer:yyy where xxx is the reasoning (thought) before generte answer to the question.
+ **final_answer**: The final answer to the question. If the question is multihop, this has the form: Summary:xxx\nAnswer:yyy Where xxx is the summary of anwers from decomposed single questions before generating final answer: yyy
+ **answer**: <i>Can ignore this field</i>
+ **meta_info**: contains the information about how the data point was created
+ **tag**: <i>can ignore this field</i>

### Generate New Training Data
We found that not much available public training data for multi-hop Q&A so we decided to create new training data using **gpt-3.5-turbo-instruct** - an OpenAI Model. Actually we create 2 kinds of multi-hop questions:

#### Multi-hop Questions asking about an attribute of 2 entities in a question

Here are some examples for these questions, **entities** are highlighted.

+ In which year were the **Seattle Public Library** and **Denver Public Library** built?
+ Is **Kailua Beach** more popular than **Waikiki Beach**?
+ How do the **Giant Anteater** and the **Lesser Anteater** differ in their reproduction processes?
  
Here is the flow to generate this kind of data:
The flow is:
+ Step 1: choose a random category (from <b>../gen_data/other_files/sub_categories.txt</b>)
+ Step 2: generate 2 entries from this category: **entity 1**, **entity 2**
+ Step 3: generate a list of **common attributes** of these 2 entities
+ Step 4: select a random attribute from the generated list --> **selected attribute**
+ Step 5: Generate **question 1** asking for the **selected attribute** of **entity 1**
+ Step 6: Generate **question 2** asking for the **selected attribute** of **entity 2**
+ Step 7: Generate **multi-hop question** that decomposed into **question 1** and **question 2**. This, for example, can be the question comparing the **selected attribute** of **entity 1** and **entity 2**
+ Step 8: Generate **paragraph 1** containing the information about the **selected attribute** of **entity 1**
+ Step 9: Generate the **reasoning 1** (thought) to answer **question 1** based on **paragraph 1**
+ Step 10: Generate the complete **answer 1** to the **question 1** based on the **reasoning 1**
+ Step 11: Generate **paragraph 2** containing the information about the **selected attribute** of **entity 2**
+ Step 12: Generate the **reasoning 2** (thought) to answer **question 2** based on **paragraph 2**
+ Step 13: Generate the complete **answer 2** to the **question 2** based on the **reasoning 2**
+ Step 14: summarize the points from **answer 1** and **answer 2** to generate the final answer to the **multi-hop question**
+ Step 15: Generate the reasoning (thought) to answer the **multi-hop question** based on the **summary**
+ Step 16: Generate the final answer to **multi-hop question** based on the reasoning

We implement this flow using the prompt: [2_entities.txt](https://github.com/khaimt/qa_expert/blob/main/gen_data/prompts/2_entities.txt) with model: **gpt-3.5-turbo-instruct**. To make the generation more creative and diverse, we used temperature=0.7 --> 1. However, we found that with these high temperatures, the step for generating answers such as step 10, 13 and 16 would be **vulerable to hallucination**. So in reality, we split the flow into 2 parts:

+ Part 1 for generating questions and paragraphs (step 1 -> 8, step 11), **temperature=0.7 -> 1**, prompt=[2_entities.txt_wo_answer.txt](https://github.com/khaimt/qa_expert/blob/main/gen_data/prompts/2_entities_wo_answer.txt)
+ Part 2 is for generating reasonings and answers (step 9, 10, 12 -> 16). Prompt=[answer_gen.txt](https://github.com/khaimt/qa_expert/blob/main/gen_data/prompts/answer_gen.txt) for step 9, 10 and step 12, 13 (generating reasonings and answers for single question 1 and single question 2). Prompt = [final_answer_gent.txt](https://github.com/khaimt/qa_expert/blob/main/gen_data/prompts/final_answer_gen.txt) for step 14, 15, 16 (generating reasonings and answers for multi-hop question). Using **temperature=0** for this part.

The script for generating data is:
```shell
python -m gen_data.gen_multi_hop_entities\
 --category-path category_path \
 --num-items-per-category 1 \
 --output-folder save_folder \
 --multi-qa-prompt gen_data/prompts/new_prompts/compare_entities_wo_answer.txt \
 --temperature 0.7 \
 --re-generate-answer
```
Please read more information about arguments in the [gen_data/gen_multi_hop_entities.py](https://github.com/khaimt/qa_expert/blob/main/gen_data/gen_multi_hop_entities.py)

#### Multi-hop Questions asking about 2 attributes of an entity in a question
## Installation
You first need to install requirements:

```
pip install -r requirements.txt
```

## Training script
### Single GPU
Here is the training script we used to fine-tune our model on 1 GPU: RTX A6000
```
python train.py \
    --model_name_or_path pretrained/Mistral-7B-v0.1  \
    --model_type mistral \
    --train_path train.json \
    --validation_path validation.json \
    --num_train_epochs 2 \
    --bf16 True \
    --per_device_train_batch_size 22 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 22 \
    --evaluation_strategy "steps" \
    --eval_steps 40 \
    --save_strategy "steps" \
    --save_steps 80 \
    --save_total_limit 5 \
    --learning_rate 4e-5 \
    --logging_steps 4 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --output_dir models/qa_v1_mistral \
    --padding longest \
    --max_sequence_length 1350 \
    --report_to wandb
```

You can use: --hf_data_path instead of --train_path and --validation_path if you want to load data from HuggingFace hub, for example:
```
--hf_data_path khaimaitien/qa-expert-multi-hop-qa-V1.0
```

Here are some notes:
+ Currently we support model_type=mistral or llama
+ In finetuning, we set max_sequence_length=1350 instead of a bigger number (2048 or even 4096) because we found that in our training data, there were only 100 data points longer than this length and bigger values of max_sequence_length would cost more memory. We also implemented a script ([find_max_token_length.py](find_max_token_length.py)) for outputing the statistis of sequence length distribution on the training data, we can base on the statistics to decide which max_sequence_length to use. Note that because we wanted to optimize the training time so we did this, if you don't need to care much about the training cost, you can just set a big number
+ When the training is completed, we need to merge the lora weights with the original weights. You can use this script: [merge_weight.py](merge_weight.py) to merge the weights
+ You can also upload your model to HuggingFace Hub using the script: [upload_model_to_hf.py](upload_model_to_hf.py)
+ Use padding=<b>longest</b> instead of <b>max_length</b> will reduce the training time considerably as we applied dynamic padding (you can take a look at: DataCollatorForMaskingLabels in train.py)

### Multiple GPU
To train on Multiple GPU, I suggest using deepspeed zero2 (note that currently, zero3 is not supported for Qlora).
You need to install deepspeed:

```
pip install deepspeed=0.11.1
```
Another note is: FSDP **doesn't work** for Lora because it requires all the parameters to be uniformly trainable or freezed. So that's why we should use Deepspeed

Here is an example:
```
deepspeed train/train.py \
    --model_name_or_path PRETRAINED_PATH  \
    --model_type llama \
    --hf_data_path khaimaitien/qa-expert-multi-hop-qa-V1.0 \
    --num_train_epochs 2 \
    --bf16 True \
    --per_device_train_batch_size 25 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 25 \
    --evaluation_strategy "steps" \
    --eval_steps 40 \
    --save_strategy "steps" \
    --save_steps 63 \
    --save_total_limit 2 \
    --learning_rate 4e-5 \
    --logging_steps 10 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --output_dir OUTPUT_DIR \
    --padding longest \
    --max_sequence_length 1350 \
    --deepspeed train/ds_config/zero2.json
```

## Evaluation
We use [HotpotQA](https://hotpotqa.github.io/) as the evaluation dataset to compare 2 models: [khaimaitien/qa-expert-7B-V1.0](https://huggingface.co/khaimaitien/qa-expert-7B-V1.0) and [khaimaitien/qa-expert-llama2-13B-V1.0](https://huggingface.co/khaimaitien/qa-expert-llama2-13B-V1.0). Actually, we randomly chose <b>500</b> examples from <b>Dev set (distractor)</b> and measure the metrics: 
+ Recall: compute the recall based on the individual words, the reason we use recall instead of F1 because the ground-truth answers are short and usually spans.
+ Accuracy of containing ground-truth: If the ground-truth answer (mostly short span) is exactly in the generated answer --> 1 else 0

Here is the result:

|Model|Recall|Accuracy of containing ground-truth|
|---|---|---|
|[khaimaitien/qa-expert-7B-V1.0](https://huggingface.co/khaimaitien/qa-expert-7B-V1.0)|0.64429|0.558|
|[khaimaitien/qa-expert-llama2-13B-V1.0](https://huggingface.co/khaimaitien/qa-expert-llama2-13B-V1.0)|0.6093|0.532|

We conclude that although the number of parameters is much smaller, <b>[khaimaitien/qa-expert-7B-V1.0](https://huggingface.co/khaimaitien/qa-expert-7B-V1.0) outperforms [khaimaitien/qa-expert-llama2-13B-V1.0](https://huggingface.co/khaimaitien/qa-expert-llama2-13B-V1.0) considerably </b>

You can run the evaluation script at the root directory of this repo:

```shell
python eval_hotpot_qa.py --model-path khaimaitien/qa-expert-7B-V1.0
```
Arguments:

+ **--model-path (str, optional)**: model to evaluate. Default="khaimaitien/qa-expert-7B-V1.0"
+ **--retriever-path (str, optional)**: The retriever model to use in evaluation. Default="intfloat/e5-base-v2"
+ **--hotpot-qa-dev-path (str, optional)**: hotpot_qa file to eval. Default="extra_data/hotpot_dev_distractor_v1_random_500.json".
+ **--inference-type (str, optional)**: type of inference, you can use Vllm to reduce the evaluation time . Default="hf"
+ **--save-path (str, optional)**: where to save the inference result, if empty, inference result is not saved. Default=""
+ **--tokenizer-path (str, optional)**: path to tokenizer, this is <b>only needed if inference_type=llama_cpp. Default=""</b>

The output of running this script will be the metrics in progress

## Training Data
You can download the training data from Huggingface Hub: [khaimaitien/qa-expert-multi-hop-qa-V1.0](https://huggingface.co/datasets/khaimaitien/qa-expert-multi-hop-qa-V1.0). 

In total, our training dataset contains 27720 data points (train & validation) including single questions and multi-hop questions. These data points are adopted and processed from available public sources or automatically generated using OpenAI Model.

### Format 
Each data point is a Json:
+ *src*: source of data point: squad.json, drop.json, boolq.json, musicque.json or gen_qa.json
+ *question*: the question, either single question or multi-hop questions
+ *inal_answer*: the final answer of the question --> model will generate this answer in the end
+ *answer*: span answer or None --> please ignore this, just an additional field of information
+ *sub_questions*: List of single questions to answer to answer the multi-hop question. If len(sub_questions) == 1 --> this is *single question*, *not multi-hop question*
    + *question*: the single question to ask
    + *answer*: the span answer of None or missing --> please ignore this, just an additional field of information
    + *long_answer*: the complete answer of this single question
    + *paragraph*: the context of the single question (this is considered as the retrieved context of the single question)
    + *unanswerable*: = True if this question is unanswerable --> you can ignore this because long_answer, note this field might be missing, default value is False.

### Single questions:
We use single questions from the following sources:
  + [Squad](https://huggingface.co/datasets/squad_v2): We randomly select 4000 answerable questions + 2400 unanswerable questions. As the answers to these questions are spans, which are short, so we use OpenAI model to generate a complete answer given the question and context. The prompt we use is: [extra_files/answer_gen.txt](../extra_files/answer_gen.txt)
  + [BoolQ](https://huggingface.co/datasets/boolq): We randomly select 1600 random questions. As the answers of these questions are yes/no, so we also use OpenAI model to generate complete answers. This type of question is more difficult and needs reasoning (like Chain-of-Thought), so we ask the model to first generate the reasoning and then the final answer. The prompt we used is: [extra_files/boolq_answer_gen.txt](../extra_files/boolq_answer_gen.txt)
  + [Drop](https://huggingface.co/datasets/drop): We randomly select 1600 random questions. The answers of these questions are also short and without explanation, so we also use OpenAI model to generate the reasoning, arithmetic sequence (if needed) and the final answer. The prompt we used is: [extra_files/drop_answer_gen.txt](../extra_files/drop_answer_gen.txt)

Here, the OpenAI models we used were: <i>gpt-3.5-turbo</i> for Squad and <i>gpt-3.5-turbo-instruct</i> for BoolQ and Drop. The temperature is always 0.

### Multi-hop questions
For multi-hop questions we use [Musique](https://github.com/StonyBrookNLP/musique) and generated data

#### [Musique](https://github.com/StonyBrookNLP/musique)
The authors built these multi-hop questions based on single questions from various sources such as squad2, natural questions, zerore, ... But we found that some single questions are not well-formed (not a question and containing: <b>">>"</b>), such as "Stadio Ciro Vigorito >> occupant". So we removed all data points that had at least one unwell-formed single question. Finally, we attained 5847 answerable multi-hop questions and we also randomly selected 2400 unanswerable multi-hop questions

Each multi-hop question is decomposed into 2 or more single questions, and each single question contains short span answer. so Here is how we process the data:
+ First, for each single question, we generate the complete answer using OpenAI model with prompt=[extra_files/answer_gen.txt](../extra_files/answer_gen.txt) (this is like handling Squad). The openAI model here is: <i>gpt-3.5-turbo</i> with temperature=0
+ Next, we generate final answer based on the <b>generated answers</b> from single questions, using prompt=[extra_files/final_answer_gen.txt](../extra_files/final_answer_gen.txt). The openAI model here is: <i>gpt-3.5-turbo-instruct</i>  with temperature=0

#### Generate training data
We used openAI model to generate multi-hop questions. The flow is:
+ Step 1: choose a random category (from <b>../extra_files/cateegories.txt</b>)
+ Step 2: generate 2 entries from this category
+ Step 3: Select an attribute of these entries to compare
+ Step 4: Generate question to compare the attributes of these 2 entries --> <b>this is the multi-hop question</b>
+ Step 5: Generate question 1 asking for the attribute of entry 1
+ Step 6: Generate question 2 asking for the attribute of entry 2
+ Step 7: Generate paragraph 1 containing the attribute of entry 1
+ Step 8: Generate paragraph 2 containing the attribute of entry 2
+ Step 9: Generate answer 1 for question 1 based on paragraph 1
+ Step 10: Generate answer 2 for question 2 based on paragraph 2
+ Step 11: Generate the thought for combining answer 1 and answer 2 to answer the <b>multi-hop question</b>
+ Step 12: Generate final answer of <b>multi-hop question</b>

This flow is implemented using the prompt: [extra_files/comparison_gen.txt](../extra_files/comparison_gen.txt), results of all steps will be generated at 1 generation from openAI model. The model we used is: <i>gpt-3.5-turbo-instruct</i>. Note that the purpose here is to generate training data, so we need the diversity, therefore we set temperature=1 instead of 0. 

Other information:
+ The purpose of choosing a random category is to diversify the training data 
+ Most of the categories from the list are from: [Sekine’s Extended Named Entities](https://nlp.cs.nyu.edu/ene/version7_1_0Beng.html)
+ We only kept the data points that totally followed the format in the prompt
+ We removed data points that 2 entries not in the generated multi-hop question.
+ We realized that because the temperature=1 so at the step 9, 10, 11, 12 the results were vulerable to hallucination, so we didn't use this result but re-generated the answers using prompts: [extra_files/answer_gen.txt](../extra_files/answer_gen.txt) and [extra_files/final_answer_gen.txt](../extra_files/final_answer_gen.txt) with temperature=0 the same way as described in handling [Musique](https://github.com/StonyBrookNLP/musique)

#### Script to generate Multi-hop training data
To generate the multi-hop training data:
```bash
export OPENAI_API_KEY=xxxx
python gen_multi_hop_qa.py --num-items-per-category 100
```
During executing this command, we can see the progress being printed out, and also see the estimated cost and time. 

At the end, we will end up having 3 files:
+ raw_multi_hop_qa.json: the generated result from 12 steps
+ filtered.json: the result after removing low-quality data points 
+ final.json: the final json file following the format of training data

Paramters:
+ --num-items-per-category: Number of items for each category. Default=100
+ --output-folder: where to save the result. Default="gen_qa"
+ --re-generate-answer: Whether or not we will generate the answers to single questions and multi-hop question, as explained above, because at first we use temperature=1 to generate so generated result is vulerable to hallucination so we should re-generate using temperature=0. Default=*False*
+ --category-path: the file containing categories, each category is separated by: "," and can be in multiple lines. Default=[extra_files/categories.txt](extra_files/categories.txt)
+ --continue-gen: If the output_folder already contains the result, whether we will continue to generate from the existing result or generate from scratch.
