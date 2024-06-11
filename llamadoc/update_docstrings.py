import sys
import json

def get_updated_docstring(code: str, docstring: str) -> str:
    code = code.replace(docstring, "")
    docstring = docstring.replace('"""', '')
    docstring = docstring.replace("'''", '')
    new_docstring = f"{docstring}\n    UPDATED"
    new_docstring = f'    """\n{new_docstring}\n    """'
    return new_docstring

def main():
    codestring = sys.argv[1]
    old_docstring = sys.argv[2]
    # print("----------------")
    # print(codestring)
    # print("----------------")
    # print(old_docstring)
    # print("----------------")
    new_docstring = get_updated_docstring(codestring, old_docstring)
    print(new_docstring)
    print(json.dumps({"new_docstring": new_docstring}))


if __name__ == "__main__":
    main()
