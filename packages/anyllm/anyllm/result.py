from types import GeneratorType
from pydantic import BaseModel

def to_result(resp: dict | BaseModel | GeneratorType) -> dict:
    """
    Convert model response to dictionary format.
    """
    if isinstance(resp, GeneratorType):
        result = {
            "content": "",
            "usage": {
                'total_tokens': 0,
                'completion_tokens': 0,
                'prompt_tokens': 0,
            },
            'finish_reason': 'unknown',
        }

        for chunk in resp:
            chunk_result = to_result(chunk)
            result['content'] += chunk_result['content']

        if chunk_result['usage']['total_tokens'] >= 0:
            result['usage']['total_tokens'] += chunk_result['usage']['total_tokens']
            result['usage']['completion_tokens'] += chunk_result['usage']['completion_tokens']
            result['usage']['prompt_tokens'] += chunk_result['usage']['prompt_tokens']
            result['finish_reason'] = chunk_result['finish_reason']
        return result


    if isinstance(resp, BaseModel):
        resp = resp.model_dump()
    elif isinstance(resp, dict):
        resp = resp
    else:
        raise ValueError("Unsupported response type")
        
    
    choice = resp['choices'][0]
    message = choice.get('message') or choice.get('delta')
    if message is None:
        raise ValueError("No message or delta found in choices")
    if not isinstance(message, dict):
        raise ValueError("Message or delta is not a dictionary")

    if 'reasoning' in message and message['reasoning']:
        content = str(message['reasoning']) + message.get('content', '')
    else:
        content = message.get('content', '') 

    usage = resp.get('usage', choice.get('usage'))
    if usage is None \
         or not isinstance(usage, dict) \
         or 'total_tokens' not in usage \
         or 'completion_tokens' not in usage \
         or 'prompt_tokens' not in usage:
        usage = {
            'total_tokens': -1,
            'completion_tokens': -1,
            'prompt_tokens': -1,
        }
    
    finish_reason = choice.get('finish_reason', 'unknown')

    res = {
        "content": content,
        "usage": {
            'total_tokens': usage['total_tokens'],
            'completion_tokens': usage['completion_tokens'],
            'prompt_tokens': usage['prompt_tokens'],
        },
        'finish_reason': finish_reason,
    }
    return res