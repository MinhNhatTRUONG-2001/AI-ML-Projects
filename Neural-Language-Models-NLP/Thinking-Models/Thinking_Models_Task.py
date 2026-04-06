# %% [markdown]
# # Thinking models task
# ### By: Nhat Truong
# ---

# %% [markdown]
# # Download packages

# %%
# %pip install python-dotenv # Needed only for local machine
%pip install torch
%pip install camel-ai rouge ollama
%pip install unsloth # For Google Colab
# %pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo # For local machine

# %%
# (FOR LOCAL MACHINE) Reinstall those packages so that it is compatible with unsloth 2026.3.3 and unsloth-zoo 2026.3.1
%pip install datasets==4.3
%pip install transformers==4.57.6
%pip install trl==0.24.0

# %% [markdown]
# # Import packages

# %%
from ollama import Client
import random
import os
import json
from dotenv import load_dotenv
from datetime import date, datetime, timedelta
from camel.datagen import CoTDataGenerator
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import GroqConfig
from camel.agents import ChatAgent

# %%
from unsloth import FastLanguageModel, is_bfloat16_supported

# %%
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer

# %% [markdown]
# # Generate questions

# %%
# Set seed for random question generation
random_seed_dayofweek = 314159265
random_seed_hexadecimal = 314159265

# %%
# Question generation functions
def question_generation_dayofweek(n_questions = 5, random_seed=None):
    datetime_range = date(9999, 12, 31) - date(1582, 10, 15) # For simplicity, we consider only the Gregorian calendar, which was used firstly in 15 Oct 1582.
    total_days = datetime_range.days

    random.seed(random_seed)

    questions_answers = {}
    for i in range(n_questions):
        # Create a random date and extract the day of week of that date
        random_days = random.randint(0, total_days)
        random_date = date(1582, 10, 15) + timedelta(days=random_days)
        day_of_week = random_date.strftime('%A')

        question = f'The day of week on {random_date.day} {random_date.strftime('%b')} {random_date.year} (Gregorian calendar) is {day_of_week}. What is the day of week on this day next year?'
        answer = date(random_date.year + 1, random_date.month, random_date.day).strftime('%A') # Day of week of the same day next year
        questions_answers[question] = answer
    return questions_answers

def question_generation_hexadecimal(n_questions = 5, random_seed=None):
    random.seed(random_seed)

    hex_digits = list('0123456789ABCDEF')
    questions_answers = {}
    for i in range(n_questions):
        hex_length = random.randint(2, 10)
        hex_number = ''.join(random.choices(hex_digits, k = hex_length))
        decimal_number = int(hex_number, base=16)

        question = f'How is this hexadecimal number "{hex_number}" converted to decimal number?'
        answer = str(decimal_number)
        questions_answers[question] = answer
    return questions_answers


# %%
questions_answers_dayofweek = question_generation_dayofweek(n_questions=10, random_seed=random_seed_dayofweek)
questions_answers_hexadecimal = question_generation_hexadecimal(n_questions=10, random_seed=random_seed_hexadecimal)
questions_answers_data = {**questions_answers_dayofweek, **questions_answers_hexadecimal}
questions_answers_data

# %% [markdown]
# # Generate COT training data

# %%
sys_msg = 'You are good at logical reasoning, critical thinking and producing step-by-step, clear and concise explanations and answers.'

# %% [markdown]
# ## Create validation datagen to compare chat answer with given correct answer

# %% [markdown]
# Groq and Llama 3.1 8b are used for the validation of generated answers. The Llama model is small and good enough for correct validation, and the rate limits of this model in Groq's free plan is 30 requests per minute and 14400 requests per day. More information about rate limits can be found here: https://console.groq.com/docs/rate-limits

# %%
# For local machine
# load_dotenv()
# groq_api_key = os.getenv("groq")

# For Google Colab
from google.colab import userdata
groq_api_key = userdata.get("groq")

os.environ["GROQ_API_KEY"] = groq_api_key

# %%
val_model = ModelFactory.create(
    model_platform=ModelPlatformType.GROQ,
    model_type=ModelType.GROQ_LLAMA_3_1_8B,
    model_config_dict=GroqConfig(temperature=0.2).as_dict(),
)
val_agent = ChatAgent(
    system_message=sys_msg,
    model=val_model,
    message_window_size=10,
)

# %%
val_datagen = CoTDataGenerator(val_agent, golden_answers=questions_answers_data)

# %% [markdown]
# ## Create COT datgen and generate answers

# %% [markdown]
# gpt-oss-120b is OpenAI's powerful open-weight reasoning model. During testing different models, I find that this model is the best one because it almost always return correct answers. It can also programmatically accessible via Ollama or some other providers.

# %% [markdown]
# ### Option 1: gpt-oss-120b & Ollama (recommended)

# %% [markdown]
# This option generates the data quite fast (about 5-30 seconds per question). I am not sure about the rate limit for the free plan, but it should be enough to generate a few tens of question-answer pairs without the rate limit issue.

# %%
# For local machine
# ollama_api_key = os.getenv("ollama")

# For Google Colab
from google.colab import userdata
ollama_api_key = userdata.get("ollama")

os.environ["OLLAMA_API_KEY"] = ollama_api_key

# %%
ollama_client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
)

# %%
cot_datagen_option = 'ollama' # For naming JSON file when saving data
generated_answers = {}
error_skip = 0 # Number of questions skipped due to exception (e.g. rate limit, empty response)

for question in questions_answers_data.keys():
    print(f"Question: {question}")

    try:
        # Get AI's thought process and answer
        answer = ollama_client.generate(
            'gpt-oss:120b-cloud',
            prompt=question,
            system=sys_msg,
            think='high', # For gpt-oss, valid values are 'low', 'medium' and 'high'
            options={
                'temperature': 0.2,
            }
        )
        generated_answers[question] = answer['response']
        print(f"AI's thought process and answer:\n{answer['response']}")

        # Verify the answer
        is_correct = val_datagen.verify_answer(question, answer['response'])
        print(f"Answer verification result: {'Correct' if is_correct else 'Incorrect'}")
        print("-" * 50)
        print()
    except Exception:
        error_skip += 1
        continue

# %% [markdown]
# ### Option 2: gpt-oss-120b, Camel AI & OpenRouter

# %% [markdown]
# In this option, the generation usually takes a bit longer time and the rate limit per day for the OpenRouter's free plan is quite low: only 50 requests/25 questions per day (More information: https://openrouter.ai/docs/api/reference/limits). However, with Camel AI, it is easier to switch to many more models and providers if needed. Also, the responses are quite more deterministic.

# %%
# For local machine
# openrouter_api_key = os.getenv("openrouter")

# For Google Colab
from google.colab import userdata
openrouter_api_key = userdata.get("openrouter")

os.environ["OPENROUTER_API_KEY"] = openrouter_api_key

# %%
# Define the chat model and chat agent
chat_model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="openai/gpt-oss-120b:free",
    url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model_config_dict={"temperature": 0.2},
)
chat_agent = ChatAgent(
    system_message=sys_msg,
    model=chat_model,
    message_window_size=10,
    token_limit=131072,
)

# %%
openrouter_datagen = CoTDataGenerator(chat_agent, golden_answers=questions_answers_data)

# %%
cot_datagen_option = 'camel-ai' # For naming JSON file when saving data
generated_answers = {}
error_skip = 0 # Number of questions skipped due to exception (e.g. rate limit, empty response)

for question in questions_answers_data.keys():
    print(f"Question: {question}")

    try:
        # Get AI's thought process and answer
        answer = openrouter_datagen.get_answer(question)
        generated_answers[question] = answer
        print(f"AI's thought process and answer:\n{answer}")

        # Verify the answer & add the question-answer pair if the answer is correct
        is_correct = val_datagen.verify_answer(question, answer)
        print(f"Answer verification result: {'Correct' if is_correct else 'Incorrect'}")
        print("-" * 50)
        print()
    except Exception:
        error_skip += 1
        continue

# %% [markdown]
# ## Transform and save generated question-answer pairs

# %%
# This function transforms the Q&A data into the Alpaca training data format, which is suitable for supervised fine-tuning (SFT).
# The transformed data is saved to a new JSON file.
def transform_qa_format(generated_qa_pairs):
    # Transform the data
    transformed_data = []
    for question, answer in generated_qa_pairs.items():
        transformed_pair = {
            "instruction": question,
            "input": "",
            "output": answer
        }
        transformed_data.append(transformed_pair)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = f'transformed-qa_{cot_datagen_option}_{timestamp}.json'

    # Write the transformed data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=4)

    return output_file, transformed_data

# %%
output_file, transformed_data = transform_qa_format(generated_answers)
print(f"Transformation complete. Output saved to: {output_file}")

# %% [markdown]
# # Configure Unsloth

# %%
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    # And also all Instruct versions and Math. Coding verisons!
    # It might take several minutes to download this model
    model_name = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# %%
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 2026,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# %%
# Presaved dataset file name
# No need to run this if the data transfromation above is done
output_file = "transformed-qa_ollama_20260310-141748.json"

# %%
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts }
pass

dataset = Dataset.from_json(output_file, split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# %% [markdown]
# # Configure and train the model

# %%
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 2026,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

# %%
trainer_stats = trainer.train()

# %% [markdown]
# # Test the fine-tuned model

# %% [markdown]
# Example questions:
# - The day of week on 31 Dec 2025 is Wednesday. What is the day of week on this day next year? (Thursday)
# - How is this hexadecimal number "1A3F" converted to decimal number? (6719)

# %%
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# %%
# alpaca_prompt is copied from above
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# Prepare the input for inference
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "The day of week on 31 Dec 2025 is Wednesday. What is the day of week on this day next year?",  # Instruction
            "",  # Input (empty for this example)
            "",  # Output (leave blank for generation)
        )
    ],
    return_tensors="pt"
).to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 4098)

# %%
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# %% [markdown]
# ## Load and use the saved model

# %%
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

if True:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# %%
#  Run the cell containing alpaca_prompt variable above first

inputs = tokenizer(
[
    alpaca_prompt.format(
        'How is this hexadecimal number "1A3F" converted to decimal number?', # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 4098)

# %% [markdown]
# # Personal comments about the testing results
# After testing the Deepseek model using around a few similar but new questions and run each question several times, I notice that:
# - Before training the model, the model has a high chance to generate wrong answers. Also, the responses are usually long and unnecessary explanation. The model usually cannot generate EOS token and runs infinitively.
# - After training the model, it may still generates incorrect results, but the chance to get good results is increase. Also, the responses are concise and there is less chance that the responses are generated infinitively.
# 
# The responses should be even better if I can use a larger model for fine-tuning. However, I cannot do it due to limited computational resources.


