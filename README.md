# QA Expert: A model for Multi-hop Question Answering

QA Expert is a Language Model (LLM) specifically fine-tuned for the task of Question Answering, with a strong emphasis on addressing Multi-hop Question Answering scenarios.

Multi-hop Question Answering is a task that necessitates the initial retrieval of multiple contexts, followed by their integration to deduce the answer to the question. For example, given the question:
"what is the population of the second biggest city in Vietnam?", the system needs to first retrieve the context for query="what is the second biggest city in Vietnam", and based on the context, the model knows that: Hanoi is the second biggest city in Vietnam, then it will need another retrieval with query=what is the population of Hanoi? --> this will give the final answer. Other examples of multi-hop question:

+ Paris and London which one is more populous?
+ How old is the president of the United States?

Please note that QA Expert has been trained not only for Multi-hop Question Answering but also for one-hop Question Answering, making it a versatile tool for addressing both types of questions.


## News

## Usage

### Model Download

### Inference 

### Server Usage


## Training
### Training Data

## To-Do

## Citation
If you feel my work is helpful, please kindly cite as:
