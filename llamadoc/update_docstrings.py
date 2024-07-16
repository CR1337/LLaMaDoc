import sys
import json
from llm_interface.llm_interface import LlmInterface

def get_updated_docstring(code: str, docstring: str) -> str:
    """
    Get the updated docstring for the given code.

    Args:
        code (str): The code string.
        docstring (str): The old docstring.

    Returns:
        str: The updated docstring.
    """
    return LlmInterface().update([code], [docstring])[0][1]


def process_docstring_update(input_data: dict) -> str:
    """
    Process and format the docstring and code, then update it.

    Args:
        input_data (dict): The input data containing the code and old docstring.

    Returns:
        str: The new docstring in JSON format.
    """
    codeString = input_data["codestring"]
    old_docstring = input_data["old_docstring"]

    start = codeString.index('"""')
    end = start + codeString[start + 1:].index('"""')
    codeString = codeString[:start] + codeString[end + 4:]

    codeString.replace('\r', '')
    old_docstring.replace('\r', '')

    new_docstring = get_updated_docstring(codeString, old_docstring)
    new_docstring = f'    """\n    {new_docstring}\n    """\n'

    return json.dumps({"new_docstring": new_docstring})


def main():
    input_data = json.loads(sys.stdin.read())
    result = process_docstring_update(input_data)
    print(result)


if __name__ == "__main__":
    main()
