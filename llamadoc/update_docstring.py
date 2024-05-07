import sys
import json

def get_updated_docstring(code: str, docstring: str) -> str:
    new_docstring = docstring.replace('"""', '')
    new_docstring = new_docstring.replace("'''", '')
    new_docstring = f"{new_docstring}\nUPDATED"
    new_docstring = f'"""\n{new_docstring}\n"""'
    return new_docstring

def main():
    json_string = "".join(sys.stdin)
    json_object = json.loads(json_string)
    code = json_object["code"]
    docstring = json_object["docstring"]
    new_docstring = get_updated_docstring(code, docstring)
    print(json.dumps({"new_docstring": new_docstring}))


if __name__ == "__main__":
    main()
