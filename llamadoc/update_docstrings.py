import sys
import json
from llm_interface.llm_interface import LlmInterface

def get_updated_docstring(code: str, docstring: str) -> str:
    # TODO: connect to backend
    
    # --- comment this out to connect to the backend ---
    # code = code.replace(docstring, "")
    # docstring = docstring.replace('"""', '')
    # docstring = docstring.replace("'''", '')
    # new_docstring = f'    """{docstring}\n    UPDATED\n    """\n'
    # return new_docstring
    # ---------------------------------------------------

    # --- uncomment this to connect to the backend ---
    return LlmInterface().update([code], [docstring])[0][1]
    # ---------------------------------------------------

def main():
    input_data = json.loads(sys.stdin.read())
    codestring = input_data["codestring"]
    old_docstring = input_data["old_docstring"]

    start = codestring.index('"""')
    end = start + codestring[start + 1:].index('"""')
    codestring = codestring[:start] + codestring[end + 4:]

    codestring.replace('\r', '')
    old_docstring.replace('\r', '')

    # with open("D:\\Programming\\update_code.txt", "w") as f:
    #     f.write(codestring)

    # with open("D:\\Programming\\update_doc.txt", "w") as f:
    #     f.write(old_docstring)



    new_docstring = get_updated_docstring(codestring, old_docstring)
    new_docstring = f'    """\n    {new_docstring}\n    """\n'

    print(json.dumps({"new_docstring": new_docstring}))


if __name__ == "__main__":
    main()
