# Training Data Creation 
Each multi-hop question can be handled by decomposing it into single questions. This datasets contains multi-hop questions and their decomposed questions. We also add single questions to this dataset to make sure that the trained model is able to handle all kinds of questions.

**You can download our training data from here:** [khaimaitien/qa-expert-multi-hop-qa-V1.0](https://huggingface.co/datasets/khaimaitien/qa-expert-multi-hop-qa-V1.0)

This dataset contains 25.5k for training and 3.19k for evaluation.

- [Training Data Creation](#training-data-creation)
  - [Format](#format)
  - [Generate New Training Data](#generate-new-training-data)
    - [Multi-hop Questions asking about an attribute of 2 entities in a question](#multi-hop-questions-asking-about-an-attribute-of-2-entities-in-a-question)
    - [Multi-hop Questions asking about 2 attributes of an entity in a question](#multi-hop-questions-asking-about-2-attributes-of-an-entity-in-a-question)
    - [Negative Paragraph Generation](#negative-paragraph-generation)
    - [Single Questions](#single-questions)
  - [Using available training datasets](#using-available-training-datasets)
  - [Lisf of Scripts for generating data](#lisf-of-scripts-for-generating-data)
    - [Script for generating sub-category from given category:](#script-for-generating-sub-category-from-given-category)
    - [Script for generating multi-hop Questions asking about an attribute of 2 entities](#script-for-generating-multi-hop-questions-asking-about-an-attribute-of-2-entities)
    - [Script for generating multi-hop Questions asking about 2 attributes of an entity in a question](#script-for-generating-multi-hop-questions-asking-about-2-attributes-of-an-entity-in-a-question)
    - [Script for generating data points with negative paragraphs](#script-for-generating-data-points-with-negative-paragraphs)
    - [Script for generating answers to the single questions and final multi-hop questions](#script-for-generating-answers-to-the-single-questions-and-final-multi-hop-questions)


## Format 
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
+ **tag**: the information about type of data, for example:
  + <i>musique-train.json</i>: train data from musique
  + <i>entities-neg_train.json</i>: data points from generating question related to 2 entities with **negative paragraph**.
  + ...


## Generate New Training Data
We found that not much available public training data for multi-hop Q&A so we decided to create new training data using **gpt-3.5-turbo-instruct** - an OpenAI Model. Actually we create 2 kinds of multi-hop questions:

### Multi-hop Questions asking about an attribute of 2 entities in a question

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

We implement this flow using the prompt: [2_entities.txt](https://github.com/khaimt/qa_expert/blob/main/gen_data/prompts/2_entities.txt) with model: **gpt-3.5-turbo-instruct**. To make the generation more creative and diverse, we used temperature=0.7 --> 1. However, we found that with these high temperatures, the step for generating answers such as step 10, 13 and 16 would be **vulerable to hallucination**. So in reality, we can split the flow into 2 parts:

+ Part 1: for generating questions and paragraphs (step 1 -> 8, step 11), **temperature=0.7 -> 1**, prompt=[2_entities.txt_wo_answer.txt](https://github.com/khaimt/qa_expert/blob/main/gen_data/prompts/2_entities_wo_answer.txt)
+ Part 2: for generating reasonings and answers (step 9, 10, 12 -> 16). Prompt=[answer_gen.txt](https://github.com/khaimt/qa_expert/blob/main/gen_data/prompts/answer_gen.txt) for step 9, 10 and step 12, 13 (generating reasonings and answers for single question 1 and single question 2). Prompt = [final_answer_gent.txt](https://github.com/khaimt/qa_expert/blob/main/gen_data/prompts/final_answer_gen.txt) for step 14, 15, 16 (generating reasonings and answers for multi-hop question). Using **temperature=0** for this part.

**Some tricks for generating a diverse dataset:**
+ The purpose of choosing a random category is to diversify the training data. At first, we manually prepared a list of [125 general categories](https://github.com/khaimt/qa_expert/blob/main/gen_data/other_files/category_list.txt) derived from [Sekine’s Extended Named Entities](https://nlp.cs.nyu.edu/ene/version7_1_0Beng.html). But we found that it was not really diverse enough, so we decided to continue to split these categories into more fine-grained categories. For each category, we used the prompt: [sub_category_gen.txt](https://github.com/khaimt/qa_expert/blob/main/gen_data/prompts/sub_category_gen.txt) to generate more fine-grained categories. Finally, we attained [3750 smaller categories](https://github.com/khaimt/qa_expert/blob/main/gen_data/other_files/sub_categories.txt) using **gpt-3.5-turbo-instruct**. You can use the script: ```python -m gen_data.gen_sub_categories``` to generate fine-grained categories from given general categories.
+ The reason why we need **step 3**: Generate a list of common attributes and then **step 4**: Select a random attribute from this list is to make the training data **more diverse**. If only step 4, I found that, for example, if category=City, the model (gpt-3.5-turbo-instruct) would choose attribute=population 90% of the times although the temperature was already **1**. 
+ During generating data, at **step 2**: generating 2 entities, I also added some randomness: {popularity_1} and {popularity_2} a random value from 1 --> 5, you can take a look at [the prompt](https://github.com/khaimt/qa_expert/blob/main/gen_data/prompts/2_entities.txt#L3C8-L3C8):
  + ```Entity 1: generate a random entity of this category such that level of popularity on internet is {popularity_1} out of 5```
  
    If the prompt was only: ```Entity 1: generate a random entity of this category``` it would only generate the popular ones even the temperature > 1. For example, for category=City in Asia, the model usually generate famous ones such as: Tokyo, Shanghai, ... instead of the less popular ones like: Da Nang, Bandung, ...
+ In the prompt, I also replaced {question_type} with a random value of: ["wh question", "yes/no question"], because I found that the model was more likely to generate Wh question other than yes/no question.

### Multi-hop Questions asking about 2 attributes of an entity in a question
Here are some examples for these questions:
+ Does Jim Gaffigan have a high net worth and is he married? 
  + **--> Entity=Jim Gaffigan; attributes: Net worth and Spouse**
+ Did the 2011 Tohoku earthquake and tsunami have a high magnitude and were there many casualties during it? 
  + **--> Entity=2011 Tohoku earthquake; attributes: Magnitude & Casualties**

For this type of questions, the flow is almost the same as the flow for generating: [Multi-hop Questions asking about an attribute of 2 entities in a question](#multi-hop-questions-asking-about-2-attributes-of-an-entity-in-a-question)

The prompts I used are: 
+ Prompt for generating all steps: [2_attributes.txt](https://github.com/khaimt/qa_expert/blob/main/gen_data/prompts/2_attributes.txt)
+ Prompt for generating generating part 1 (questions and paragraphs): [2_attributes_wo_answer.txt](https://github.com/khaimt/qa_expert/blob/main/gen_data/prompts/2_entities_wo_answer.txt). I used the same prompt for generating part 2 (reasonings and answers)
  
### Negative Paragraph Generation
Besides generating paragraphs that contain the answer to the single questions, I also generate data points that the paragraphs don't contain the answer. I called these: **negative paragraphs**. I randomly picked 1200 data points from each multi-hop entities questions and multi-hop attributes questions to generate data points with **negative paragraphs**

Assume that the chosen data point is: x = (e1, a1, q1, p1, e2, a2, q2, p2) where: 
+ e1: entity 1, a1: attribute 1, q1: single question 1; p1: paragraph of single question 1 containing information about attribute a1 of e1.
+ e2: entity 2, a2: attribute 2, q2: single question 2; p2: paragraph of single question 2 containing information about attribute a2 of e2.

If x is from 2 entities data, a1 = a2 = selected attribute; e1 != e2
If x is from 2 attributes data, e1 = e2 = selected entity, a1 != a2

To create negative context data, I randomly picked one of 3 options:
+ replace q1 with a **negative paragraph**
+ replace q2 with a **negative paragraph**
+ replace both q1 and q2 with new **negative paragraph**

For example, if we want to generate a **negative paragraph** from original paragraph written for entity: **e** and attribute **a**. We first create a prompt for generating paragraph for an entity and an attribute (this prompt is: [gen_paragraph](https://github.com/khaimt/qa_expert/blob/main/gen_data/prompts/gen_paragraph.txt)) then we use this prompt to generate negative paragraph by:
+ replacing **e** with another new entity of the same category, using prompt: [gen_new_entity.txt](https://github.com/khaimt/qa_expert/blob/main/gen_data/prompts/gen_new_entity.txt)
+ replacing **a** with another new attribute from attribute list generated at generating data
+ Or replacing both **e** and **a** with new entity and new attribute

For example, we have the original paragraph for entity="Tokyo", attribute="GDP". We can generate a negative paragraph by using the prompt: [gen_paragraph](https://github.com/khaimt/qa_expert/blob/main/gen_data/prompts/gen_paragraph.txt) with 
+ entity="Shanghai", attribute="GDP" --> replacing entity
+ entity="Tokyo", attribute="attraction" --> replacing attribute
+ entity="Shanghai", attribute="attraction" --> replacing both entity and attribute

When we replacing the original paragraphs with negative paragraphs, we need to update the answers to single questions and final answer to multi-hop question using prompts: [answer_gen.txt](https://github.com/khaimt/qa_expert/blob/main/gen_data/prompts/answer_gen.txt) and [final_answer_gen.txt](https://github.com/khaimt/qa_expert/blob/main/gen_data/prompts/answer_gen.txt) as desribed in Part 2 of previous section.

You can see the script for generating negative-paragraphs data points in the section [List of Scripts](#lisf-of-scripts)

### Single Questions
To create data points that the question is a single question instead of multi-hop question (field ``multihop=False``), we just used the single questions in multi-hop questions

## Using available training datasets
We found that [Musique](https://github.com/StonyBrookNLP/musique) is the most suitable dataset for multi-hop Q&A for brideging entity so we made use of this. Here are the steps we process this dataset:
+ Remove data points containing single questions that are not well-formed (containing: **">>"**), such as: "Stadio Ciro Vigorito >> occupant"
+ For each single question, we generated the **complete answers** because this dataset only contains span answer for questions. Complete answers of single questions were generated using prompt: [answer_gen.txt](https://github.com/khaimt/qa_expert/blob/main/gen_data/prompts/answer_gen.txt) and complete answers of final questions were generated using prompt: [final_answer_gent.txt](https://github.com/khaimt/qa_expert/blob/main/gen_data/prompts/final_answer_gen.txt)
+ Remove data points that generated **complete answers** don't contain the span answer

You can find data points of this dataset by finding ones that whose field ``tag`` contains: **"musique"**


## Lisf of Scripts for generating data

### Script for generating sub-category from given category:
```shell 
python -m gen_data.gen_sub_categories \
 --category-path gen_data/other_files/category_list.txt \
 --save_path Where_to_save.json
```

### Script for generating multi-hop Questions asking about an attribute of 2 entities

**Note that to run the script, you need to set the OPENAI_API_KEY first by:**
```shell
export OPENAI_API_KEY=YOUR_KEY_HERE
```

Example:
```shell
python -m gen_data.gen_multi_hop_entities\
 --category-path gen_data/other_files/sub_category_gen.txt \
 --num-items-per-category 1 \
 --output-folder save_folder/entities \
 --multi-qa-prompt gen_data/prompts/new_prompts/2_entities_wo_answer.txt.txt \
 --temperature 0.7 \
 --re-generate-answer
```
Please read more information about arguments in the [gen_data/gen_multi_hop_entities.py](https://github.com/khaimt/qa_expert/blob/main/gen_data/gen_multi_hop_entities.py)

### Script for generating multi-hop Questions asking about 2 attributes of an entity in a question
Example:
```shell
 python -m gen_data.gen_multi_hop_attributes\
 --category-path gen_data/other_files/sub_category_gen.txt \
 --num-items-per-category 1 \
 --output-folder save_folder/attributes \
 --multi-qa-prompt gen_data/prompts/new_prompts/2_attributes_wo_answer.txt \
 --temperature 0.7 \
  --re-generate-answer
```
### Script for generating data points with negative paragraphs
Example:
```shell
python -m gen_data.gen_negatives \
--input-path multi-hop_data_path \
--save-path JSON_RESULT_FILE.json \
--gen_num 1200 \
```
+ multi-hop_data_path: is the json file of the generated data 
  
### Script for generating answers to the single questions and final multi-hop questions
This script will fill in ``long_answer`` and ``final_answer`` in the input_json file
Example:
```shell
python -m gen_data.gen_answer \
  --input-path: input_path \
  --output-path: output_path
```
+ input_path: the Json file containing data points that field ``long_answer`` in "sub_questions" is Null