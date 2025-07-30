import pandas as pd

# Sample data simulating the RMF rules input
data = {
    'Action ID': ['GV-1.1-001', 'GV-1.2-001'],
    'Suggested Action': [
        'Align GAI development and use with applicable laws and regulations, including those related to data privacy, copyright and intellectual property law.',
        'Establish transparency policies and processes for documenting the origin and history of training data and generated data …'
    ],
    'GAI Risks': [
        'Data Privacy; Harmful Bias & Homogenization; Intellectual Property',
        'Data Privacy; Information Integrity; Intellectual Property'
    ]
}

# Load into DataFrame
df = pd.DataFrame(data)

# Define keyword mapping for system layers
layer_keywords = {
    'Model': ['training data', 'model', 'fine-tuning', 'robustness', 'enhancement'],
    'Output': ['output', 'content', 'bias', 'toxicity', 'misinformation', 'disinformation'],
    'User': ['stakeholder', 'user', 'governance', 'civil rights', 'human-AI']
}

# Function to assign system layer
def assign_system_layer(text):
    text = text.lower()
    for layer, keywords in layer_keywords.items():
        if any(keyword in text for keyword in keywords):
            return layer
    return 'Unclassified'

# Apply the function to the GAI Risks and Suggested Action fields
df['System Layer (Auto)'] = df.apply(
    lambda row: assign_system_layer(row['GAI Risks'] + ' ' + row['Suggested Action']),
    axis=1
)

import ace_tools as tools 

# Display the DataFrame to the user
tools.display_dataframe_to_user(name="AI System Layer Assignment", dataframe=df) 
