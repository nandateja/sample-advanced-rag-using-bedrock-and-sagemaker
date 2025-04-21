from typing import Dict, Any

def print_citations(response: Dict[str, Any]) -> None:
    """
    Extract and print citations used in the generated response
    
    Parameters:
    - response (Dict[str, Any]): Response from Bedrock retrieve_and_generate
    """
    try:
        # Extract the response text
        output_text = response['output']['text']
        print("\nGenerated Response:")
        print("=" * 80)
        print(output_text)
        print("=" * 80)
        
        # Extract citations
        if 'citations' in response and response['citations']:
            retrieved_references = response['citations'][0]['retrievedReferences']
            print(f"\nNumber of citations: {len(retrieved_references)}")
            
            for i, reference in enumerate(retrieved_references, 1):
                print(f"\nCitation {i}:")
                print("-" * 40)
                
                # Print content
                if 'content' in reference and 'text' in reference['content']:
                    text = reference['content']['text']
                    print(f"Content: {text[:300]}..." if len(text) > 300 else f"Content: {text}")
                
                # Print location
                if 'location' in reference:
                    print(f"Source: {reference['location']}")
                
                # Print metadata
                if 'metadata' in reference:
                    print("Metadata:")
                    for key, value in reference['metadata'].items():
                        print(f"  - {key}: {value}")
        else:
            print("\nNo citations available in the response.")
    
    except Exception as e:
        print(f"Error extracting citations: {str(e)}")