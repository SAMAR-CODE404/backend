import logging
from datetime import datetime
from typing import Callable, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_node(node_name):
    def decorator(func):
        def wrapper(*args, **kwargs): 
            logger.info(f"Executing node: {node_name}")
            start_time = datetime.now()
            result = func(*args, **kwargs)  
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Completed node: {node_name} in {duration:.2f} seconds")
            return result
        return wrapper
    return decorator


def truncate_text(text, max_length=10000):
    if len(text) <= max_length:
        return text
    return text[:max_length] + "... [truncated]"