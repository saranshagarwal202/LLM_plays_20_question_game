# 20 Questions Game using Reinforcement Learning

This project implements an AI agent that learns to play the 20 Questions game through Reinforcement Learning. The agent, based on Meta's Llama 3 8B model, learns to ask intelligent questions and make informed guesses to identify words chosen from a predefined set of places (countries, cities, and landmarks).

## Project Overview

In the classic 20 Questions game, players take turns trying to guess each other's chosen word by asking up to 20 yes/no questions. This implementation uses Reinforcement Learning, specifically the Proximal Policy Optimization (PPO) algorithm, to train an AI model to become proficient at this game.

### Key Components

1. **Base Model**: Meta's Llama 3 8B Instruct model
2. **Training Method**: Proximal Policy Optimization (PPO)
3. **Model Optimization**: LoRA (Low-Rank Adaptation) for efficient fine-tuning
4. **Environment**: Custom environment implementing the 20 Questions game mechanics

## Technical Implementation

### Environment

The environment (`Environment` class) manages the game state and interactions:

- Maintains the game state including questions asked and answers received
- Handles word selection from the keywords dataset
- Processes questions and generates appropriate yes/no responses
- Tracks the number of questions asked and validates game rules

### Reward System

The reward function is designed to encourage:
- Quick correct guesses (higher rewards for fewer questions)
- Penalize incorrect guesses after 20 questions
- Provide scaled rewards based on the number of questions used:
  ```python
  r = [15.5, 15.25, 14.975, ..., 2.71, 1.181]  # Decreasing rewards for more questions
  ```

### Training Architecture

The model uses several optimization techniques:
- **LoRA Configuration**:
  ```python
  lora_config = LoraConfig(
      r=16,
      lora_alpha=32,
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM",
      target_modules=['q_proj', 'v_proj']
  )
  ```

- **PPO Configuration**:
  ```python
  config = PPOConfig(
      model_name=model_id,
      learning_rate=1.41e-5,
      batch_size=8,
      mini_batch_size=1,
      optimize_device_cache=True,
  )
  ```

### Data Generation

The training process uses a custom `DataGenerator` class that:
- Creates training samples with varying question counts
- Handles tokenization and formatting of inputs
- Manages batching for efficient training

## Setup and Requirements

### Dependencies

```
trl[peft]
bitsandbytes
loralib
transformers
torch
numpy
datasets
tqdm
```

### Dataset

The project requires a `keywords.json` file containing:
- Words to be guessed (places, landmarks, countries)
- Categories for each word
- Alternative names/spellings where applicable

## Training Process

The training loop:
1. Generates batches of game scenarios
2. Runs PPO optimization steps
3. Computes rewards based on model performance
4. Updates model parameters
5. Saves the trained model periodically

## Usage

To train the model:
```python
# Initialize environment and trainer
dataset = DataGenerator(100)
ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    dataset=dataset,
    tokenizer=tokenizer
)

# Run training
for epoch in range(epochs):
    for batch in ppo_trainer.dataloader:
        # Generate responses
        response_tensors = [ppo_trainer.generate(x, **generation_kwargs) 
                          for x in query_tensors]
        
        # Compute rewards and update model
        rewards = reward(batch['query'], batch['response'], batch['metadata'])
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
```

## Model Outputs

The trained model can:
1. Ask relevant yes/no questions to narrow down possibilities
2. Make educated guesses based on received answers
3. Learn from feedback through the reward system

## Future Improvements

Potential areas for enhancement:
1. Implementing more sophisticated reward mechanisms
2. Using Different models.
3. Optimizing the question-asking strategy
4. Improving the guessing mechanism