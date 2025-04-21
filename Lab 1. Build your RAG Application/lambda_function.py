import json
from abc import abstractmethod, ABC
from typing import List
from urllib.parse import urlparse
import boto3
import logging

# Initialize logging for debugging purposes
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Abstract class for chunking strategy
class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        # This method should be implemented by subclasses to define the chunking logic
        raise NotImplementedError()

# Simple implementation of the Chunker class that splits text into chunks of 100 words
class SimpleChunker(Chunker):
    def chunk(self, text: str) -> List[str]:
        words = text.split()  # Split the text into words
        # Return chunks of 100 words each
        return [' '.join(words[i:i+100]) for i in range(0, len(words), 100)]

# Lambda function handler
def lambda_handler(event, context):
    logger.debug('input={}'.format(json.dumps(event)))  # Log the incoming event for debugging
    s3 = boto3.client('s3')  # Create an S3 client for interacting with S3

    # Extract the list of input files and the bucket name from the event
    input_files = event.get('inputFiles')
    input_bucket =  event.get('bucketName')

    # Check if the necessary inputs are provided in the event
    if not all([input_files, input_bucket]):
        raise ValueError("Missing required input parameters")  # Raise an error if any input is missing
    
    output_files = []  # List to store output file information
    chunker = SimpleChunker()  # Initialize the chunker object for chunking content

    # Process each input file
    for input_file in input_files:
        content_batches = input_file.get('contentBatches', [])  # Get the content batches from the input file
        file_metadata = input_file.get('fileMetadata', {})  # Extract file metadata
        original_file_location = input_file.get('originalFileLocation', {})  # Extract original file location

        processed_batches = []  # List to store information of processed batches

        # Process each content batch
        for batch in content_batches:
            input_key = batch.get('key')  # Get the key (location) of the batch file in S3

            # Check if the key is missing and raise an error if so
            if not input_key:
                raise ValueError("Missing uri in content batch")

            # Read file content from S3 using the read_s3_file function
            file_content = read_s3_file(s3, input_bucket, input_key)
            
            # Process content by chunking it using the chunker
            chunked_content = process_content(file_content, chunker)
            
            # Create an output key for the processed file (write to S3)
            output_key = f"Output/{input_key}"
            
            # Write the chunked content back to S3
            write_to_s3(s3, input_bucket, output_key, chunked_content)
            
            # Add the processed batch information to the list
            processed_batches.append({
                'key': output_key
            })
        
        # Prepare the output file information, including original location, metadata, and processed batches
        output_file = {
            'originalFileLocation': original_file_location,
            'fileMetadata': file_metadata,
            'contentBatches': processed_batches
        }
        output_files.append(output_file)
    
    # Return the result containing all processed output files
    result = {'outputFiles': output_files}
    return result

# Function to read a file from S3
def read_s3_file(s3_client, bucket, key):
    response = s3_client.get_object(Bucket=bucket, Key=key)  # Get the file from S3
    return json.loads(response['Body'].read().decode('utf-8'))  # Read and decode the file content as JSON

# Function to write content to S3
def write_to_s3(s3_client, bucket, key, content):
    s3_client.put_object(Bucket=bucket, Key=key, Body=json.dumps(content))  # Upload the content to S3

# Function to process content and chunk it
def process_content(file_content: dict, chunker: Chunker) -> dict:
    chunked_content = {
        'fileContents': []  # Initialize the result to store chunked content
    }
    
    # Process each content item in the file content
    for content in file_content.get('fileContents', []):
        content_body = content.get('contentBody', '')  # Extract the content body (text)
        content_type = content.get('contentType', '')  # Extract the content type
        content_metadata = content.get('contentMetadata', {})  # Extract metadata for the content
        
        words = content['contentBody']  # Get the body of the content (string of words)
        chunks = chunker.chunk(words)  # Use the chunker to split the content into chunks
        
        # Add each chunk to the chunked content list
        for chunk in chunks:
            chunked_content['fileContents'].append({
                'contentType': content_type,
                'contentMetadata': content_metadata,
                'contentBody': chunk  # The chunked content
            })
    
    return chunked_content  # Return the chunked content
