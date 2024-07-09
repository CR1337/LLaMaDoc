import sys
import json
from llm_interface.llm_interface import LlmInterface

def get_updated_docstring(code: str, docstring: str) -> str:
    return LlmInterface().update([code], [docstring])[0][1]


def main():
    input_data = json.loads(sys.stdin.read())
    codeString = input_data["codestring"]
    old_docstring = input_data["old_docstring"]

    start = codeString.index('"""')
    end = start + codeString[start + 1:].index('"""')
    codeString = codeString[:start] + codeString[end + 4:]

    codeString.replace('\r', '')
    old_docstring.replace('\r', '')


    new_docstring = get_updated_docstring(codeString, old_docstring)
    new_docstring = f'    """\n    {new_docstring}\n    """\n'

    print(json.dumps({"new_docstring": new_docstring}))


if __name__ == "__main__":
    main()
