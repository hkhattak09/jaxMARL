"""Response parser for LLM-generated code.

Extracts Python code and JSON from LLM responses,
validates syntax, and extracts function definitions.
"""

import re
import ast
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class ParsedFunction:
    """Container for a parsed function."""
    name: str
    source: str
    args: List[str]
    docstring: Optional[str]


def parse_code_blocks(text: str, language: str = "python") -> List[str]:
    """Extract code blocks of a specific language from text.
    
    Args:
        text: Text containing code blocks
        language: Language identifier (python, json, etc.)
        
    Returns:
        List of code block contents
    """
    # Pattern matches ```language ... ```
    pattern = rf"```{language}\s*\n?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    # Clean up whitespace
    return [m.strip() for m in matches if m.strip()]


def parse_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON block from text.
    
    Args:
        text: Text containing JSON code block
        
    Returns:
        Parsed JSON as dict, or None if not found/invalid
    """
    import json
    
    blocks = parse_code_blocks(text, "json")
    
    for block in blocks:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    
    return None


def extract_functions(code: str) -> Dict[str, ParsedFunction]:
    """Extract function definitions from Python code.
    
    Args:
        code: Python source code
        
    Returns:
        Dict mapping function name to ParsedFunction
    """
    functions = {}
    
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Invalid Python syntax: {e}")
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Get function source
            # We need to extract from original code since ast doesn't preserve formatting
            func_source = _extract_function_source(code, node.name)
            
            # Get argument names
            args = [arg.arg for arg in node.args.args]
            
            # Get docstring
            docstring = ast.get_docstring(node)
            
            functions[node.name] = ParsedFunction(
                name=node.name,
                source=func_source,
                args=args,
                docstring=docstring,
            )
    
    return functions


def _extract_function_source(code: str, func_name: str) -> str:
    """Extract a function's source code from a larger code block.
    
    Args:
        code: Full source code
        func_name: Name of function to extract
        
    Returns:
        Function source code
    """
    lines = code.split('\n')
    in_function = False
    func_lines = []
    base_indent = 0
    
    for line in lines:
        # Check for function definition
        stripped = line.lstrip()
        if stripped.startswith(f"def {func_name}("):
            in_function = True
            base_indent = len(line) - len(stripped)
            func_lines.append(line)
            continue
        
        if in_function:
            # Check if we're still in the function
            if line.strip() == "":
                func_lines.append(line)
            elif line.startswith(' ' * (base_indent + 1)) or line.startswith('\t'):
                func_lines.append(line)
            elif stripped.startswith("def ") or stripped.startswith("class "):
                # New definition at same level = end of function
                break
            elif len(line) - len(stripped) <= base_indent and stripped:
                # Non-empty line at same or lower indent = end of function
                break
            else:
                func_lines.append(line)
    
    return '\n'.join(func_lines)


def validate_generated_code(code: str, required_functions: List[str] = None) -> Tuple[bool, str]:
    """Validate generated Python code.
    
    Args:
        code: Python source code to validate
        required_functions: List of function names that must be present
        
    Returns:
        (is_valid, error_message)
    """
    if required_functions is None:
        required_functions = ["compute_reward", "compute_prior_policy"]
    
    # Check syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    
    # Extract functions
    try:
        functions = extract_functions(code)
    except ValueError as e:
        return False, str(e)
    
    # Check required functions exist
    for func_name in required_functions:
        if func_name not in functions:
            return False, f"Missing required function: {func_name}"
    
    return True, ""


def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse complete LLM response.
    
    Extracts:
    - Python code blocks (reward and policy functions)
    - JSON metadata (sub-goals, capabilities)
    - Reasoning text
    
    Args:
        response: Raw LLM response text
        
    Returns:
        Dict with 'code', 'functions', 'metadata', 'reasoning'
    """
    result = {
        "code": None,
        "functions": {},
        "metadata": None,
        "reasoning": None,
        "raw_response": response,
    }
    
    # Extract reasoning section
    reasoning_match = re.search(
        r"### Reasoning:?\s*\n(.*?)(?=### Code|```python)",
        response,
        re.DOTALL | re.IGNORECASE
    )
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()
    
    # Extract Python code
    code_blocks = parse_code_blocks(response, "python")
    if code_blocks:
        # Combine all Python blocks (in case split across multiple)
        combined_code = "\n\n".join(code_blocks)
        result["code"] = combined_code
        
        # Extract functions
        try:
            result["functions"] = extract_functions(combined_code)
        except ValueError as e:
            print(f"Warning: Could not extract functions: {e}")
    
    # Extract JSON metadata
    result["metadata"] = parse_json_block(response)
    
    return result


def get_function_source(parsed: Dict[str, Any], func_name: str) -> Optional[str]:
    """Get source code for a specific function from parsed response.
    
    Args:
        parsed: Result from parse_llm_response
        func_name: Name of function to get
        
    Returns:
        Function source code, or None if not found
    """
    if func_name in parsed.get("functions", {}):
        return parsed["functions"][func_name].source
    return None
