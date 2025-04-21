import boto3
from pprint import pprint
import pandas as pd

def list_bedrock_models():
    try:
        # Create a Bedrock client
        bedrock = boto3.client('bedrock')
        
        # List all foundation models
        response = bedrock.list_foundation_models()
        
        # Filter for enabled models
        enabled_models = [
            model for model in response['modelSummaries'] 
            if model['modelLifecycle']['status'] == 'ACTIVE'
        ]
        df = pd.DataFrame(enabled_models)
        pd.set_option('display.max_rows', df.shape[0]+1)
        return df
    except Exception as e:
        print(f"Error occurred: {str(e)}")