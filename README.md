# QA Expert: LLM for Multi-hop Question Answering

QA Expert is a Language Model (LLM) specifically fine-tuned for the task of Question Answering, with a strong emphasis on addressing Multi-hop Question Answering scenarios.

<p align="center">
  <img src="assets/1-shot-example.jpg", width="300", height="250">
</p>
<p align="center">
An example of 1-shot question and how QA Expert LLM handle multi-hop Q&A</p>

<p align="center">
  <img src="assets/hotpot_qa_bridge.jpg", width="300", height="250">
  <img src="assets/hotpot_qa_compare.jpg", width="300", height="250">
</p>
<p align="center">
Examples of 2-shot questions and how QA Expert LLM handle multi-hop Q&A. The left is an example of bridging entitiy and the right is an example of comparing entities</p>

Multi-hop Question Answering is a task that necessitates the retrieval of multiple contexts, followed by their integration to deduce the answer to the question. 

QA Expert will analyze the question and the retrieval result to decide whether additional retrieval is required or if the answer should be generated. So the output of QA expert is either <b>calling retrieval function with a query </b> or <b>generating the final answer</b>, this is a little bit similar to the output of OpenAI function calling: call a function or generate text response.

## News
- [2023/10/12] We released our finetuned model: <b>khaimaitien/qa-expert-7B-V1.0</b>based on [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) + our training data: [khaimaitien/qa-expert-multi-hop-qa-V1.0](https://huggingface.co/datasets/khaimaitien/qa-expert-multi-hop-qa-V1.0)

## Content
1. [Usage](#usage)
2. [Training data](#training-data)
3. [Training](#training)
4. [Citation](#citation)

## Usage
### Model Download

| Size | Hugging Face Repo |
| ---  | --- |
| 7B | [khaimaitien/qa-expert-7B-V1.0](https://huggingface.co/khaimaitien/qa-expert-7B-V1.0) | 

### Inference 
As our LLM will 

### Server Usage


## Training Data
You can download the training data from Huggingface Hub: [khaimaitien/qa-expert-multi-hop-qa-V1.0](https://huggingface.co/datasets/khaimaitien/qa-expert-multi-hop-qa-V1.0). 

In total, our training dataset contains 22720 data points including single questions and multi-hop questions.

#### Single questions:
  + [Squad](https://huggingface.co/datasets/squad_v2): 4000 randomly answerable questions + 2400 unanswerable questions. As the answers to these questions are spans, short, so we use OpenAI model to generate a complete answer given the question and context. The prompt we use is: <b>extra_files/answer_gen.txt</b>. 
  + [BoolQ](https://huggingface.co/datasets/boolq): 1600 random questions. As the answers of these questions are yes/no, so we also use OpenAI model to generate complete answers. This type of question is more difficult and needs reasoning (like Chain-of-Thought), so we ask ChatGPT to first generate the reasoning and then the final answer. The prompt we used is: 
  + Drop: 1600 random questions. The answers of these questions are also short and without explanation, so we also use OpenAI model to generate the reasoning and the final answer. The prompt we used is:

Here, the OpenAI model we used is: gpt-3.5-turbo and temperature=0 (greedy)

#### Multi-hop questions
For multi-hop questions we use [Musique](https://github.com/StonyBrookNLP/musique) and generated data

##### [Musique](https://github.com/StonyBrookNLP/musique)
This contains: 5847 answerable multi-hop questions and 2400 unanswerable multi-hop questions. The authors built these multi-hop questions based on single questions from various sources such as squad2, natural questions, zerore, ... But we found that some single questions are not well-formed (not a question and containing: <b>">>"</b>), such as "Stadio Ciro Vigorito >> occupant". So we removed all data points that had at least one unwell-formed single question.

Each multi-hop question is decomposed into 2 or more single questions, and each single question contains short span answer. so Here is how we process the data:
+ First, for each single question, we generate the complete answer using OpenAI model with prompt=<b>extra_files/answer_gen.txt</b> (this is like handling Squad). The openAI model here is: <i>gpt-3.5-turbo</i> with temperature=0
+ Next, we generate final answer based on the <b>generated answers</b> from single questions, using prompt=<b>extra_files/final_answer_gen.txt</b>. The openAI model here is: <i>gpt-3.5-turbo-instruct</i>  with temperature=0

##### Generate training data
We used openAI model to generate multi-hop questions. The flow is:
+ Step 1: choose a random category (from <b>extra_files/cateegories.txt</b>)
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

This flow is implemented using the prompt: <b>extra_files/comparison_gen.txt</b>, results of all steps will be generated at 1 generation from openAI model. The model we used is: <i>gpt-3.5-turbo-instruct</i>. Note that the purpose here is to generate training data, so we need the diversity, so I set temperature=1 instead of 0. 

Other information:
+ The purpose of choosing a random category is to diversify the training data 
+ Most of the categories from the list are from: [Sekine’s Extended Named Entities](https://nlp.cs.nyu.edu/ene/version7_1_0Beng.html)
+ We only kept the data points that totally followed the format in the prompt
+ We removed data points that 2 entries not in the generated multi-hop question.
+ We realized that because the temperature=1 so at the step 9, 10, 11, 12 the results were vulerable to hallucination, so we didn't use this result but re-generated the result using prompts: <b>extra_files/answer_gen.txt</b> and <b>extra_files/final_answer_gen.txt</b> with temperature=0 the same way as described in handling [Musique](https://github.com/StonyBrookNLP/musique)

## Training


## To-Do

## Citation
If you feel my work is helpful, please kindly cite as:
```bibtex
@Misc{qa-expert,
      title={QA Expert: LLM for Multi-hop Question Answering},
      author={Khai Mai},
      howpublished={\url{https://github.com/khaimt/qa_expert}},
      year={2023},
}
```