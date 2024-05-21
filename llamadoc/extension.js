const vscode = require('vscode');
const os = require('os');
const { exec } = require('child_process');

const LLAMADOC_UPDATE_DOCSTRING_COMMAND = 'llamadoc.updateDocstringCommand';
const LLAMADOC_DISMISS_COMMAND = 'llamadoc.dismissCommand';
const LLAMADOC_SCAN_FILE = 'llamadoc.scanFile';

let decorationTypes = [];
let executedActions = new Set(); // Track executed actions
let codeActionProvider = null; // Track the provider registration
let lightbulbLines = []; // Track lines for which lightbulbs are registered
let dismissAllButton; // Track the dismiss all button

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
    registerCommands(context);
    createStatusBarButton(context);
}

function deactivate() {
    if (codeActionProvider) {
        codeActionProvider.dispose();
    }
}

function registerCommands(context) {
    context.subscriptions.push(
        vscode.commands.registerCommand(LLAMADOC_UPDATE_DOCSTRING_COMMAND, (lineNumber) => {
            clearSpecificDecoration(lineNumber);
            executedActions.add(lineNumber);
            vscode.window.showInformationMessage('Updated Docstring.');
        }),
        vscode.commands.registerCommand(LLAMADOC_DISMISS_COMMAND, (lineNumber) => {
            clearSpecificDecoration(lineNumber);
            executedActions.add(lineNumber);
            vscode.window.showInformationMessage('Dismissed Docstring Update.');
        }),
        vscode.commands.registerCommand(LLAMADOC_SCAN_FILE, scanFile),
        vscode.commands.registerCommand('llamadoc.dismissAll', dismissAllExpiredDocstrings)
    );
}

function createStatusBarButton(context) {
    const searchButton = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    searchButton.text = "$(search) Search for Out of Date Docstrings";
    searchButton.command = LLAMADOC_SCAN_FILE;
    searchButton.show();
    context.subscriptions.push(searchButton);

    dismissAllButton = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 99);
    dismissAllButton.text = "$(close) Dismiss All Expired Docstrings";
    dismissAllButton.command = 'llamadoc.dismissAll';
    context.subscriptions.push(dismissAllButton);
}

function scanFile() {
    executedActions.clear(); // Clear the set to allow new actions
    lightbulbLines = []; // Clear previous lightbulb lines

    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) {
        vscode.window.showErrorMessage('No active editor found.');
        return;
    }

    const fileName = activeEditor.document.fileName;
    const platform = os.platform();
    const pyCommand = platform === 'win32' 
        ? `python -u ${__dirname}\\find_docstrings.py "${fileName}"`
        : `python3 -u ${__dirname}/find_docstrings.py "${fileName}"`;

    exec(pyCommand, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error: ${error.message}`);
            return;
        }
        if (stderr) {
            console.error(`stderr: ${stderr}`);
        }
        console.log(`stdout: ${stdout}`);
        const jsonObject = JSON.parse(stdout);
        spawnButtons(jsonObject);

        dismissAllButton.show();
    });
}

function getUpdatedText() {
    return "    \"\"\"This is the updated docstring\"\"\"\n";
}

function registerCodeActionsProvider() {
    if (codeActionProvider) {
        codeActionProvider.dispose();
    }

    codeActionProvider = vscode.languages.registerCodeActionsProvider('python', {
        provideCodeActions(document, range, context, token) {
            let actions = [];
            lightbulbLines.forEach(({ lineNumber, docstring_start, docstring_end, replacementText }) => {
                if (executedActions.has(lineNumber)) {
                    return;
                }

                if (range.start.line === lineNumber && range.end.line === lineNumber) {
                    const updateAction = createCodeAction('Update Docstring', LLAMADOC_UPDATE_DOCSTRING_COMMAND, vscode.CodeActionKind.RefactorRewrite, [lineNumber]);
                    const dismissAction = createCodeAction('Dismiss', LLAMADOC_DISMISS_COMMAND, vscode.CodeActionKind.Refactor, [lineNumber]);

                    const start = new vscode.Position(docstring_start - 1, 0);
                    const end = new vscode.Position(docstring_end, 0);
                    const docstringRange = new vscode.Range(start, end);

                    updateAction.edit = new vscode.WorkspaceEdit();
                    updateAction.isPreferred = true;
                    dismissAction.isPreferred = true;
                    updateAction.edit.replace(document.uri, docstringRange, replacementText);
                    
                    actions.push(updateAction, dismissAction);
                }
            });
            return actions;
        }
    }, { providedCodeActionKinds: [vscode.CodeActionKind.Refactor, vscode.CodeActionKind.RefactorRewrite] });
}

function spawnLightbulbs(lineNumber, docstring_start, docstring_end, replacementText) {
    lightbulbLines.push({ lineNumber, docstring_start, docstring_end, replacementText });
    registerCodeActionsProvider();
}

function createCodeAction(title, command, kind, args = []) {
    const action = new vscode.CodeAction(title, kind);
    action.command = { command, title, arguments: args };
    return action;
}

function spawnButtons(jsonObject) {
    clearDecorations();
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    jsonObject.forEach(pythonFunc => {
        if (!pythonFunc.has_docstring || pythonFunc.up_to_date) return;

        const lineNumber = pythonFunc.start_line - 1;
        const buttonDecorationType = vscode.window.createTextEditorDecorationType({
            after: {
                contentText: "Expired Docstring!",
                color: '#f0f0f0',
                backgroundColor: '#333333',
                margin: '0 0 0 2em',
                border: '1px solid #444444',
                fontSize: '12px'
            }
        });

        const startPosition = new vscode.Position(lineNumber, 75);
        const decoration = { range: new vscode.Range(startPosition, startPosition) };
        editor.setDecorations(buttonDecorationType, [decoration]);
        
        const replacementText = getUpdatedText();
        spawnLightbulbs(lineNumber, pythonFunc.docstring_start_line, pythonFunc.docstring_end_line, replacementText);

        decorationTypes.push({ decorationType: buttonDecorationType, lineNumber });
    });
}

function clearDecorations() {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
        decorationTypes.forEach(({ decorationType }) => {
            editor.setDecorations(decorationType, []);
            decorationType.dispose();
        });
        decorationTypes = [];
    }
}

function clearSpecificDecoration(lineNumber) {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
        decorationTypes = decorationTypes.filter(({ decorationType, lineNumber: decLineNumber }) => {
            if (decLineNumber === lineNumber) {
                editor.setDecorations(decorationType, []);
                decorationType.dispose();
                return false;
            }
            return true;
        });

        lightbulbLines = lightbulbLines.filter(lightbulb => lightbulb.lineNumber !== lineNumber);
        registerCodeActionsProvider();
    }
}

function dismissAllExpiredDocstrings() {
    clearDecorations();
    lightbulbLines = [];
    registerCodeActionsProvider();
    dismissAllButton.hide(); // Geschmackssache
    vscode.window.showInformationMessage('Dismissed all expired docstrings.');

}

module.exports = {
    activate,
    deactivate
};
