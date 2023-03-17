from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
from flask import Flask, request, jsonify, render_template

# Load the trained GPT model and tokenizer
model_path = 'gpt2_model/'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Define a function to generate Jira tickets from user inputs
def generate_ticket(input_text):
    # Preprocess the input text
    input_text = re.sub(r'\W+', ' ', input_text).strip().lower()
    # Encode the input text as input IDs
    input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors='pt')
    # Generate output IDs using the GPT model
    output_ids = model.generate(input_ids, max_length=1024, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    # Decode the output IDs as text
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Parse the output text to extract the Jira ticket fields
    ticket = {'summary': '', 'description': '', 'issue_type': ''}
    for line in output_text.split('\n'):
        if line.startswith('summary:'):
            ticket['summary'] = line[8:].strip()
        elif line.startswith('description:'):
            ticket['description'] = line[12:].strip()
        elif line.startswith('issue_type:'):
            ticket['issue_type'] = line[11:].strip()
    return ticket

# Define a Flask app with a single route that handles POST requests
app = Flask(__name__)
@app.route('/', methods=['POST'])
def create_ticket():
    # Get the user input from the request body
    input_text = request.json['input_text']
    # Generate a Jira ticket from the user input
    ticket = generate_ticket(input_text)
    # Return the Jira ticket as JSON
    return jsonify(ticket)

# Define a simple HTML template for the user interface
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Jira Ticket Generator</title>
</head>
<body>
    <h1>Jira Ticket Generator</h1>
    <form method="POST" action="/">
        <label for="input_text">Enter your request:</label><br>
        <textarea id="input_text" name="input_text" rows="5" cols="50"></textarea><br>
        <input type="submit" value="Create Ticket">
    </form>
</body>
</html>
"""

# Run the Flask app
if __name__ == '__main__':
    app.run()
