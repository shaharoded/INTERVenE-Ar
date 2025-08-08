import os

# Go two levels up from this config file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Data file paths (relative to project root)
TRAIN_TEMPORAL_DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'train', 'synthetic_diabetes_temporal_data.csv')
TRAIN_CTX_DATA_FILE      = os.path.join(PROJECT_ROOT, 'data', 'train', 'synthetic_diabetes_context_data.csv')
TEST_TEMPORAL_DATA_FILE  = os.path.join(PROJECT_ROOT, 'data', 'test', 'synthetic_diabetes_temporal_data.csv')
TEST_CTX_DATA_FILE       = os.path.join(PROJECT_ROOT, 'data', 'test', 'synthetic_diabetes_context_data.csv')

# Define the prediction targets and <eot> tokens to terminate the inference
OUTCOMES = [
    "KETOACIDOSIS",
    "KIDNEY_DISORDER",
    "COMA",
    "EYE_DISORDER",
    "HYPOGLYCEMIA",
    "HYPERGLYCEMIA",
    "CARDIOVASCULAR_DISORDER",
    "INFECTION",
    "NEUROVASCULAR_COMPLICATION"
]

ADMISSION_TOKEN = "ADMISSION"
DEATH_TOKEN = "DEATH"
RELEASE_TOKEN = "RELEASE"

TERMINAL_OUTCOMES = [RELEASE_TOKEN, DEATH_TOKEN]

MEAL_TOKENS = ["MEAL_Breakfast", "MEAL_Lunch", "MEAL_Dinner", "MEAL_Night"] # Keep ordered! concept_value tokens