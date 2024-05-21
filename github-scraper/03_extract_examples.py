import os
from dataclasses import dataclass
from typing import Any, List, Dict
import json
import ast
from difflib import SequenceMatcher
from tqdm import tqdm

@dataclass
class Function:
    code_similarity: float
    docstring_similarity: float
    commit: str
    date: str
    code: str
    docstring: str
    code_updated: bool
    docstring_updated: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code_similarity": self.code_similarity,
            "docstring_similarity": self.docstring_similarity,
            "commit": self.commit,
            "date": self.date,
            "code": self.code,
            "docstring": self.docstring,
            "code_updated": self.code_updated,
            "docstring_updated": self.docstring_updated
        }

@dataclass
class ExampleFile:
    repo_index: str
    file_path: str
    functions: Dict[str, List[Function]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_index": self.repo_index,
            "file_path": self.file_path,
            "functions": {
                k: [f.to_dict() for f in v] 
                for k, v in self.functions.items()
            }
        }


class DataExtractor:

    REPO_PATH: str = "extracted-py-files"
    EXAMPLE_FILE_PATH: str = "extracted-examples.json"
    INDENTATION = ' ' * 4

    def _is_trivial_function(self, node):
        # Filter out any docstrings at the start of the body
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Str):
            body = body[1:]

        # Now check if the remaining body contains only `pass` or `...`
        if len(body) == 1 and isinstance(body[0], (ast.Pass, ast.Expr)):
            if isinstance(body[0], ast.Pass):
                return True
            if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and body[0].value.value is Ellipsis:
                return True
        return False

    def _extract_functions(
        self, item: Dict[str, int | str | List[Dict[str, str]]]
    ) -> ExampleFile | None:
        repo_index = item['repo_index']
        file_path = item['file_path']

        functions: Dict[str, List[Function]] = {}

        for version in item['versions']:
            commit = version['commit']
            date = version['date']

            full_code = (
                version['blob']
                    .replace('\r\n', '\n')
                    .replace('\r', '\n')
                    .replace('\t', self.INDENTATION)
            )
            full_code_lines = full_code.split('\n')

            if len(full_code_lines) < 3:
                continue
            while len(full_code_lines) >= 3:
                try:
                    tree = ast.parse("\n".join(full_code_lines))
                except (SyntaxError, IndentationError) as exception:
                    line_number = exception.lineno
                    if line_number > len(full_code_lines) // 2:
                        full_code_lines = full_code_lines[:line_number-1]
                    else:
                        full_code_lines = full_code_lines[line_number:]
                else:
                    break

            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                if self._is_trivial_function(node):
                    continue
                docstring = ast.get_docstring(node)
                if not docstring:
                    continue

                function_name = node.name

                start_line = node.lineno
                end_line = node.end_lineno

                function_code_lines = full_code_lines[start_line-1:end_line]
                if len(function_code_lines) == 0:
                    continue
                while all(l.startswith(self.INDENTATION) or l.isspace() for l in function_code_lines):
                    function_code_lines = [l[len(self.INDENTATION):] for l in function_code_lines]

                function_code = "\n".join(function_code_lines)
                docstring_delimiter = '"""' if '"""' in function_code else "'''"
                try:
                    docstring_start = function_code.index(docstring_delimiter)
                except ValueError:
                    continue
                docstring_end = function_code[docstring_start+1:].index(docstring_delimiter)
                function_code = function_code[:docstring_start] + function_code[docstring_start+1+docstring_end+3:]

                if function_name in functions:
                    last_function_code = functions[function_name][-1].code
                    last_docstring = functions[function_name][-1].docstring

                    code_updated = last_function_code != function_code
                    docstring_updated = last_docstring != docstring

                    if not code_updated and not docstring_updated:
                        continue

                    code_similarity = SequenceMatcher(a=last_function_code, b=function_code).ratio()
                    docstring_similarity = SequenceMatcher(a=last_docstring, b=docstring).ratio()
                else:
                    code_similarity = None
                    docstring_similarity = None
                    code_updated = None
                    docstring_updated = None
                    functions[function_name] = []

                function = Function(
                    code_similarity=code_similarity,
                    docstring_similarity=docstring_similarity,
                    commit=commit,
                    date=date,
                    code=function_code,
                    docstring=docstring,
                    code_updated=code_updated,
                    docstring_updated=docstring_updated
                )
                functions[function_name].append(function)

        if len(functions) == 0:
            return None
        
        return ExampleFile(
            repo_index=repo_index,
            file_path=file_path,
            functions=functions
        )


    def extract_examples(self):
        number_of_files = sum(
            len(os.listdir(os.path.join(self.REPO_PATH, directory))) 
            for directory in os.listdir(self.REPO_PATH)
        )
        
        example_files = []

        with tqdm(total=number_of_files, desc="Extracting examples") as pbar:
            for directory in os.listdir(self.REPO_PATH):
                for file in os.listdir(os.path.join(self.REPO_PATH, directory)):
                    with open(os.path.join(self.REPO_PATH, directory, file)) as file:
                        item = json.load(file)
                        example_file = self._extract_functions(item)
                        if example_file is not None:
                            example_files.append(example_file)
                    pbar.update(1)
    
        with open(self.EXAMPLE_FILE_PATH, 'w') as file:
            json.dump([ef.to_dict() for ef in example_files], file, indent=4)


if __name__ == "__main__":
    extractor = DataExtractor()
    extractor.extract_examples()
