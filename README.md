# Train Large Language Models on chat data. 

LLaMA-7b trained and verified, performs on-par/better(in most cases) than most Huggingface variants on similar model train on cleaned alpaca data.

See inference notebook for some examples how it compares against the LLaMA-7b model.

## Settting up Container 

```sh
cd dockercontext
make build 
```

Then enter the container 

```sh
make enter
```

## Setting up datasets

Cleaned alpaca data is used to train the model, but any data on a similar format will suffice. 

```python
python src/setup.py --local --model 'decapoda-research/llama-7b-hf' --data_file data/alpaca_data_cleaned.json --max_tokens 512
```

if ```--local``` is not set, the dataset will get uploaded to Azure ML together with the tokenizer model. ```--model``` accepts Huggingface decoder models. 

## Memory requirements 

For LLAMA 7 billion model 

| Scenario                     | Model    | Precision | Batch Size | DeepSpeed Stage | Memory (GB) |
|------------------------------|----------|-----------|------------|-----------------|-------------|
| Inference                    | Llama 7b | 16-bit    | 1          | N/A             | 13          |
| Inference                    | Llama 7b | 8-bit     | 1          | N/A             | 8           |
| Train                        | Llama 7b | 16-bit    | 2          | N/A             | > 49        |
| Train                        | Llama 7b | 16-bit    | 2          | 2               | 27          |
| Train                        | Llama 7b | 16-bit    | 4          | 2               | 38          |
| Train                        | Llama 7b | 16-bit    | 4          | 2               | 43          |
| Train                        | Llama 7b | 16-bit    | 4          | 3               | 45          |
| Train                        | Llama 7b | 16-bit    | 6          | 2               | > 49        |

## Training 

```python
python src/train_lightning.py --config config.yaml --local
```

if ```--local``` we will upload the finished model to Azure ML

config.yaml contains all options that controls the model training, from deepspeed config options to model options and flags for doing LORA-training. 

## With Reacher Libary

Set up the Reacher instance, connecting to the remote machine running the docker images in dockercontext. 

```python
from reacher.reacher import Reacher, ReacherDocker, RemoteClient
from reacher.reacher import create_notebook, create_tensorboard

reacher = Reacher(
    build_name="training_alpacha_lora",
    host="",
    port=8961,
    user="root",
    ssh_key_filepath=".ssh/id_rsa",
    prefix_cmd="PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home//.local/bin"
)
reacher.setup()
```

Create a tensorboard session on the remote and do port-forwarding between the remote and your local machine

```python
create_tensorboard(reacher, remote_port=55069, local_port=55069, logdir=reacher.build_path+"/artifacts/runs")
```

A notebook server can also be setup on the remote, with port-forwarding to your local machine

```python
create_notebook(reacher, remote_port=55068, local_port=55068)
```

```python
# Process the data
reacher.execute(
    "python src/setup.py --local --model 'decapoda-research/llama-7b-hf' --data_file alpaca_data_cleaned.json --max_tokens 512",
    context=["src", "data/alpaca_data_cleaned.json"],
    wrap_in_screen=True,
    named_session="setup",
)
# Train the model
reacher.execute(
    "python src/train_lightning.py --config config.yaml --local",
    context=["src", "config.yaml"],
    wrap_in_screen=True,
    named_session="training",
)
```

Then Grab the checkpoint
```python
reacher.get("<path>/final.ckpt", "weights/")
```

# Inference 

Load the config used during training, set 8 bit mode to be true for the inference. 

```python
conf = yaml.safe_load(Path("config.yaml").read_text())
conf["model"]["load_in_8bit"] = True
conf["model"]["device_map"] = "auto"
```

Setup the model

```python
model = GeneratorModel(**conf["model"], torch_dtype=torch.float16)
model.load_lora_pretrained("weights/alpacha_lora/adapter_model.bin")
```

```python
for text in texts:
    response = generate_text(
        model=model,
        tokenizer=tokenizer,
        text=text,
        max_length=150,
        device="cuda",
    )
    print(response+ "\n\n" +  "-" * 40 +  "\n\n")
```

Clearly there are some hallucinations..

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Think of a way to use the following material to build a shelter.

### Input:
sticks and a piece of cloth

### Response:You can create a simple lean-to by using sticks as poles and wrapping them around a tree or other structure. The stick will hold up the roof while the cloth will provide protection from rain and wind.

----------------------------------------

Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
How can you stack the objects in a stable manner?

### Input:
a book, 9 eggs, a nail, a plant, a shoe

### Response:The egg should be placed on top of the book and then the shoes could go next to it. The plant would need to be placed at the bottom of the pile. Then the nails could be used as support for the whole structure by placing them underneath the base of the plant. This way they will provide stability without being seen or touching any part of the object.

----------------------------------------

Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Which is the third largest city in Sweden, list the first and second as well

### Response:The third largest city in Sweden is Gothenburg (Göteborg). The first two are Stockholm and Malmö.

----------------------------------------

Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Write a poem describing my love for my wife

### Input:
We meet 10 years ago 

### Response:My life has been filled with joy since I met you so long ago
I've found my soul mate, our bond strong as steel
Our hearts are one, forevermore
Through thick and thin, through it all
The world can never separate us two
She's always there to comfort me
And she'll be by my side till death do us part
Love of this kindredominever
Above the stars above
Brighting brightest star
Shining on high
Inspiring me to fly higher than ever before
Without her I would have no where to go
Her heart is mine, and mine hers
Foreverlasting
No matter what comes or

----------------------------------------

Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Who is the prime minister in Sweden?

### Response:The Prime Minister of Sweden is Stefan Löfven, who has been serving since 2014. He was appointed by King Carl XVI Gustaf.

----------------------------------------

Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Make up a game for two people that likes math and physics, make some examples as well

### Response:A fun game to play with someone who enjoys math and physics could be called "The Physics Game". In this game you have to answer questions about physics such as how far away from Earth would you need to travel in order to reach light speed? The first person to get all of their answers correct wins! Another example of a game like this one can be called “Physics Puzzle” where players must solve puzzles based on physics principles. For instance, they are given a picture of a car and then asked what type of engine it has or which way does its wheels move when it turns left or right. This game requires knowledge of basic physics concepts but also problem solving skills. 2 more games like these include “Physics

----------------------------------------
```