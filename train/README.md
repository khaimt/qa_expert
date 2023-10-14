# Fine-tuning
We chose to use Qlora fine-tuning to reduce the memory usage. Our model ([khaimaitien/qa-expert-multi-hop-qa-V1.0](https://huggingface.co/datasets/khaimaitien/qa-expert-multi-hop-qa-V1.0)) was trained on <b>1 RTX A6000</b> from [vast.ai](vast.ai). It took around <b>12 hours</b> for 2 epoch on our training data.

## Content
- [Fine-tuning](#fine-tuning)
  - [Content](#content)
  - [Installation](#installation)
  - [Training script](#training-script)
  - [Training Data](#training-data)
    - [Single questions:](#single-questions)
    - [Multi-hop questions](#multi-hop-questions)
      - [Musique](#musique)
      - [Generate training data](#generate-training-data)
      - [Format](#format)

## Installation
You first need to install requirements:

```
pip install -r requirements.txt
```

## Training script
Here is the training script we used to fine-tune our model on 1 RTX A6000
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

Here are some notes:
+ Currently we support model_type=mistral or llama
+ In finetuning, we set max_sequence_length=1350 instead of a bigger number (2048 or even 4096) because we found that in our training data, there were only 100 data points longer than this length and bigger values of max_sequence_length would cost more memory. We also implemented a script ([find_max_token_length.py](find_max_token_length.py)) for outputing the statistis of sequence length distribution on the training data, we can base on the statistics to decide which max_sequence_length to use. Note that because we wanted to optimize the training time so we did this, if you don't need to care much about the training cost, you can just set a big number
+ When the training is completed, we need to merge the lora weights with the original weights. You can use this script: [merge_weight.py](merge_weight.py) to merge the weights
+ You can also upload your model to HuggingFace Hub using the script: [upload_model_to_hf.py](upload_model_to_hf.py)
+ Use padding=<b>longest</b> instead of <b>max_length</b> will reduce the training time considerably as we applied dynamic padding (you can take a look at: DataCollatorForMaskingLabels in train.py)
+ Currently, the training script only works on a single GPU, we will fix the training script so that it will be able to train on multiple GPU in the future

## Training Data
You can download the training data from Huggingface Hub: [khaimaitien/qa-expert-multi-hop-qa-V1.0](https://huggingface.co/datasets/khaimaitien/qa-expert-multi-hop-qa-V1.0). 

In total, our training dataset contains 27720 data points (train & validation) including single questions and multi-hop questions. These data points are adopted and processed from available public sources or automatically generated using OpenAI Model.

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

This flow is implemented using the prompt: [extra_files/comparison_gen.txt](../extra_files/comparison_gen.txt), results of all steps will be generated at 1 generation from openAI model. The model we used is: <i>gpt-3.5-turbo-instruct</i>. Note that the purpose here is to generate training data, so we need the diversity, so I set temperature=1 instead of 0. 

Other information:
+ The purpose of choosing a random category is to diversify the training data 
+ Most of the categories from the list are from: [Sekine’s Extended Named Entities](https://nlp.cs.nyu.edu/ene/version7_1_0Beng.html)
+ We only kept the data points that totally followed the format in the prompt
+ We removed data points that 2 entries not in the generated multi-hop question.
+ We realized that because the temperature=1 so at the step 9, 10, 11, 12 the results were vulerable to hallucination, so we didn't use this result but re-generated the result using prompts: [extra_files/answer_gen.txt](../extra_files/answer_gen.txt) and [extra_files/final_answer_gen.txt](../extra_files/final_answer_gen.txt) with temperature=0 the same way as described in handling [Musique](https://github.com/StonyBrookNLP/musique)

#### Format 
Each data point is a Json:
+ *src*: source of data point: squad.json, drop.json, boolq.json, musicque.json or gen_qa.json
+ *question*: the question, either single question or multi-hop questions
+ *inal_answer*: the final answer of the question --> model will generate this answer in the end
+ *answer*: span answer or None --> please ignore this, just an additional field of information
+ *sub_questions*: List of single questions to answer to answer the multi-hop question. If len(sub_questions) == 1 --> this is single question, not multi-hop question
    + *question*: the single question to ask
    + *answer*: the span answer of None or missing --> please ignore this, just an additional field of information
    + *long_answer*: the complete answer of this single question
    + *paragraph*: the context of the single question (this is considered as the retrieved context of the single question)
    + *unanswerable*: = True if this question is unanswerable --> you can ignore this because long_answer, note this field might be missing, default value is False.