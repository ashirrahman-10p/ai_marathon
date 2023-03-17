import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load the preprocessed data
dataset = TextDataset(
    tokenizer=GPT2Tokenizer.from_pretrained('gpt2'),
    file_path='stackoverflow_preprocessed.txt',
    block_size=128
)

# Set up the model and training parameters
model = GPT2LMHeadModel.from_pretrained('gpt2')
training_args = TrainingArguments(
    output_dir='./results',          
    overwrite_output_dir=True,       
    num_train_epochs=5,              
    per_device_train_batch_size=16,  
    save_steps=1000,                 
    save_total_limit=2,              
    prediction_loss_only=True,      
)

# Set up the trainer
trainer = Trainer(
    model=model,                      
    args=training_args,              
    data_collator=DataCollatorForLanguageModeling(tokenizer=dataset.tokenizer, mlm=False),
    train_dataset=dataset           
)

# Start training
trainer.train()
