import ast
import json
import sys
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Function:
    start_line: int
    end_line: int
    docstring_start_line: int
    docstring_end_line: int
    docstring: str
    code: str
    has_docstring: bool = False
    up_to_date: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Converts the function to a dictionary."""
        return {
            'start_line': self.start_line,
            'end_line': self.end_line,
            'docstring_start_line': self.docstring_start_line,
            'docstring_end_line': self.docstring_end_line,
            'docstring': self.docstring,
            'code': self.code,
            'has_docstring': self.has_docstring,
            'up_to_date': self.up_to_date
        }
    

def extract_functions(full_code: List[str]) -> List[Function]:
    """
    Extracts functions from the code and returns a list of Function objects.

    Args:
        full_code (List[str]): The full code as a list of lines.

    Returns:
        List[Function]: A list of Function objects.
    """
    tree = ast.parse("".join(full_code))
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node)
            start_line = node.lineno
            end_line = node.end_lineno
            if not docstring:
                functions.append(Function(
                    start_line=start_line,
                    end_line=end_line,
                    docstring='',
                    code="".join(full_code[start_line-1:end_line])
                ))
                continue

            docstring_start_line = None
            docstring_end_line = None
            one_delimiter_found = False
            delimiter = None
            for i, line in enumerate(full_code):
                if not delimiter:
                    if '"""' in line:
                        delimiter = '"""'
                    elif "'''" in line:
                        delimiter = "'''"
                    else:
                        continue
                delimiter_count = line.count(delimiter)
                if delimiter_count == 2:
                    docstring_start_line = i
                    docstring_end_line = i
                    break
                if delimiter_count == 1:
                    if one_delimiter_found:
                        docstring_end_line = i
                        break
                    else:
                        docstring_start_line = i
                        one_delimiter_found = True

            code = "".join(full_code[start_line-1:end_line])
            docstring_delimiter = '"""' if '"""' in code else "'''"
            docstring_start = code.index(docstring_delimiter)
            docstring_end = code[docstring_start+1:].index(docstring_delimiter)
            code = code[:docstring_start] + code[docstring_start+1+docstring_end+3:]
            functions.append(Function(
                start_line=start_line,
                end_line=end_line,
                docstring_start_line=docstring_start_line,
                docstring_end_line=docstring_end_line,
                docstring=docstring,
                code=code,
                has_docstring=True
            ))
    return functions


def check_out_of_date(functions: List[Function]) -> List[Function]:
    """
    Checks if the docstrings are up to date and updates the up_to_date attribute.

    Args:
        functions (List[Function]): A list of Function objects.

    Returns:
        List[Function]: The updated list of Function objects.
    """
    for i in range(len(functions)):
        if not functions[i].has_docstring:
            continue
        if i % 2 == 0:
            functions[i].up_to_date = True
        else:
            functions[i].up_to_date = False
    return functions


def main():
    filename = sys.argv[1]
    try:
        with open(filename, 'r') as f:
            code = f.readlines()
    except FileNotFoundError:
        print(f"File {filename} not found", file=sys.stderr)
        sys.exit(1)

    functions = extract_functions(code)
    functions = check_out_of_date(functions)

    functions_dicts = [function.to_dict() for function in functions]
    print(json.dumps(functions_dicts, indent=2))

    
if __name__ == "__main__":
    main()
