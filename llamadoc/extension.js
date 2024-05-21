const path = require('path');
const vscode = require('vscode');
const os = require('os');

/**
 * @param {vscode.ExtensionContext} context
 */

let decorationTypes = []; // Declare decorationTypes at the top level

vscode.commands.registerCommand('llamadoc.myCommand', () => {
    vscode.window.showInformationMessage('My Command was invoked!');
    //clearDecorations();
});

function getUpdatedText(linenumber) {
    let newDocstring = "    \"\"\"This is the updated docstring\"\"\"\n";
    //let lines = $(newDocstring).val().split("\n");  
    return newDocstring;
}

// TODO: update next linenumber when changing amount of docstring lines
// TODO: dismiss expired tag (button)
// TODO: remove tag when updated docstring

function spawnLightbulbs(lineNumber, docstring_start, docstring_end, replacementText) {
    vscode.languages.registerCodeActionsProvider('python', {
        provideCodeActions(document, range, context, token) {
            if (range.start.line === lineNumber && range.end.line === lineNumber) {
                const action = new vscode.CodeAction('Update Docstring', vscode.CodeActionKind.QuickFix);
                action.edit = new vscode.WorkspaceEdit();
                action.diagnostics = [];
                action.isPreferred = true;
                action.command = {
                    command: 'llamadoc.myCommand',
                    title: 'Update Docstring'
                };

                const start = new vscode.Position(docstring_start - 1, 0);
                const end = new vscode.Position(docstring_end, 0);
                const docstringRange = new vscode.Range(start, end);

                action.edit.replace(document.uri, docstringRange, replacementText);
                return [action];
            }
            return [];
        }
    });
}

function spawnButtons(jsonObject) {
    clearDecorations(); // Clear previous decorations
    for (const pythonFunc of jsonObject) {
        if (!pythonFunc.has_docstring || pythonFunc.up_to_date) continue;
        const lineNumber = pythonFunc.start_line - 1;

        let buttonDecorationType = vscode.window.createTextEditorDecorationType({
            after: {
                //contentIconPath: vscode.Uri.file(path.join(__dirname, 'images', 'hour-glass.png')),
                contentText: "Expired Docstring!",
                color: '#f0f0f0',  // Text color
                backgroundColor: '#333333',
                margin: '0 0 0 2em',
                border: '1px solid #444444',  // Slightly lighter border color
                fontSize: '12px'
            }
        });

        let editor = vscode.window.activeTextEditor;
        if (editor) {
            let startPosition = new vscode.Position(lineNumber, 75);
            let endPosition = new vscode.Position(lineNumber, 75);

            let decoration = {
                range: new vscode.Range(startPosition, endPosition)
            };
            let docstring_start = pythonFunc.docstring_start_line;
            let docstring_end = pythonFunc.docstring_end_line;
            let replacementText = getUpdatedText(lineNumber);
            
            spawnLightbulbs(lineNumber, docstring_start, docstring_end, replacementText);

            editor.setDecorations(buttonDecorationType, [decoration]);
            // Store the decoration type so it can be disposed of later
            decorationTypes.push(buttonDecorationType);
        }
    }
}

function clearDecorations() {
    let editor = vscode.window.activeTextEditor;
    if (editor) {
        for (const decorationType of decorationTypes) {
            // Remove the decoration by setting an empty array
            editor.setDecorations(decorationType, []);
            // Dispose of the decoration type
            decorationType.dispose();
        }
        // Clear the array
        decorationTypes = [];
    }
}

function activate(context) {

    let button = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    button.text = "$(search)  Search for Out of Date Docstrings";
    button.command = "llamadoc.scanFile";
    button.show();

    let disposable = vscode.commands.registerCommand('llamadoc.scanFile', function () {
        
        const { exec } = require('child_process');
        const argument = vscode.window.activeTextEditor.document.fileName;

        let windowsSystem = `python -u ${__dirname}\\find_docstrings.py "${argument}"`;
        let otherSystem = `python3 -u ${__dirname}/find_docstrings.py "${argument}"`;

        let pyCommand;
        
        const platform = os.platform();
        
        if (platform == 'win32') {
            pyCommand = windowsSystem;
        } else {
            pyCommand = otherSystem;
        }

        exec(pyCommand, (error, stdout, stderr) => {
            if (error) {
                console.error(`Error: ${error.message}`);
                return;
            }
            if (stderr) {
                console.error(`stderr: ${stderr}`);
            }
            console.log(`stdout: ${stdout}`);
            let jsonObject = JSON.parse(stdout);
            spawnButtons(jsonObject);
        });
    });

    context.subscriptions.push(disposable);

}

function deactivate() {}

module.exports = {
    activate,
    deactivate
}
