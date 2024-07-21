# Architecture

```mermaid
flowchart LR
    vscode[VSCode Extension]
    subgraph Backend
        subgraph Docker
            fastapi[FastAPI]
            codegemma[codegemma-2b]
            codebert[codebert-base]
        end
    end

    vscode --- fastapi
    fastapi --- codegemma
    fastapi --- codebert
```
