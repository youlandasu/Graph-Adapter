from prompt_files.prompts_config import GRAPH_PROMPT_TOKENS

GRAPH_CONFIG = {
    'ratio' : 0.8,
    'graph_layers': 3, 
    'slot_size': len(GRAPH_PROMPT_TOKENS),
    'undirected': True,
    'self_loops': True,
    'graph_heads': 1,
    'graph_drop': 0.0,
}