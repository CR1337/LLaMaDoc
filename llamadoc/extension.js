const vscode = require('vscode');
const os = require('os');
const { exec, spawn } = require('child_process');

const LLAMADOC_UPDATE_DOCSTRING_COMMAND = 'llamadoc.updateDocstringCommand';
const LLAMADOC_DISMISS_COMMAND = 'llamadoc.dismissCommand';
const LLAMADOC_SCAN_FILE = 'llamadoc.scanFile';

let decorationTypes = [];
let executedActions = new Set();
let codeActionProvider = null;
let lightbulbLines = [];
let dismissAllButton;
let jsonObject;

function activate(context) {
    registerCommands(context);
    createStatusBarButton(context);
}

function deactivate() {
    codeActionProvider?.dispose();
}

function showLoadingMessage(message) {
    const loadingMessage = vscode.window.setStatusBarMessage(`$(sync~spin) ${message}`);
    return loadingMessage;
}

function registerCommands(context) {
    context.subscriptions.push(
        vscode.commands.registerCommand(LLAMADOC_UPDATE_DOCSTRING_COMMAND, updateDocstring),
        vscode.commands.registerCommand(LLAMADOC_DISMISS_COMMAND, dismissDocstring),
        vscode.commands.registerCommand(LLAMADOC_SCAN_FILE, scanFile),
        vscode.commands.registerCommand('llamadoc.dismissAll', dismissAllExpiredDocstrings)
    );
}

function createStatusBarButton(context) {
    const searchButton = createButton("$(search) Search for Out of Date Docstrings", LLAMADOC_SCAN_FILE, vscode.StatusBarAlignment.Left, 100);
    context.subscriptions.push(searchButton);

    dismissAllButton = createButton("$(close) Dismiss All Expired Docstrings", 'llamadoc.dismissAll', vscode.StatusBarAlignment.Left, 99);
    context.subscriptions.push(dismissAllButton);
}

function createButton(text, command, alignment, priority) {
    const button = vscode.window.createStatusBarItem(alignment, priority);
    button.text = text;
    button.command = command;
    button.show();
    return button;
}

async function updateDocstring(lineNumber, docstringStartLine, docstringEndLine, start_line, end_line) {
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) {
        vscode.window.showErrorMessage('No active editor found.');
        return;
    }

    const document = activeEditor.document;
    const oldDocstring = document.getText(new vscode.Range(new vscode.Position(docstringStartLine - 1, 0), new vscode.Position(docstringEndLine, 0)));
    const codestring = document.getText(new vscode.Range(new vscode.Position(start_line - 1, 0), new vscode.Position(end_line, 0)));
    
    const loadingMessage = showLoadingMessage('Updating docstring...');
    try {
        const newDocstring = await getUpdatedText(codestring, oldDocstring);
        const newDocstringLength = newDocstring.split('\n').length - 1
        const oldDocstringLength = oldDocstring.split('\n').length - 1  
        const lineDiff = newDocstringLength - oldDocstringLength
    
        clearSpecificDecoration(lineNumber);
        executedActions.add(lineNumber);
        vscode.window.showInformationMessage('Updated Docstring.');
    
        const edit = new vscode.WorkspaceEdit();
        const editRange = new vscode.Range(new vscode.Position(docstringStartLine - 1, 0), new vscode.Position(docstringEndLine, 0));
        edit.replace(document.uri, editRange, newDocstring);
        await vscode.workspace.applyEdit(edit);
    
        if (newDocstringLength !== oldDocstringLength) {
            updateLineDifference(lineNumber + 1,  lineDiff);
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Error updating docstring: ${error.message}`);
    } finally {
        loadingMessage.dispose();
    }
}

function dismissDocstring(lineNumber) {
    clearSpecificDecoration(lineNumber);
    executedActions.add(lineNumber);
    vscode.window.showInformationMessage('Dismissed Docstring Update.');
}

function scanFile() {
    executedActions.clear();
    lightbulbLines = [];
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) {
        vscode.window.showErrorMessage('No active editor found.');
        return;
    }

    const loadingMessage = showLoadingMessage('Scanning for out-of-date docstrings...');
    const pyCommand = getPythonCommandFind(activeEditor.document.fileName);
    exec(pyCommand, (error, stdout, stderr) => {
        loadingMessage.dispose();
        if (error) {
            console.error(`Error: ${error.message}`);
            return;
        }
        if (stderr) {
            console.error(`stderr: ${stderr}`);
        }
        jsonObject = JSON.parse(stdout);
        spawnButtons(jsonObject);
        dismissAllButton.show();
    });
}

function getPythonCommandFind(fileName) {
    const platform = os.platform();
    const scriptPath = platform === 'win32' ? `${__dirname}\\find_docstrings.py` : `${__dirname}/find_docstrings.py`;
    const pythonCommand = platform === 'win32' ? 'python -u' : 'python3 -u';
    return `${pythonCommand} ${scriptPath} "${fileName}"`;
}

function getPythonCommandUpdate(codestring, oldDocstring) {
    const platform = os.platform();
    const scriptPath = platform === 'win32' ? `${__dirname}\\update_docstrings.py` : `${__dirname}/update_docstrings.py`;
    const pythonCommand = platform === 'win32' ? 'python' : 'python3';
    return [pythonCommand, '-u', scriptPath];
}

async function getUpdatedText(codestring, oldDocstring) {
    const pyCommand = getPythonCommandUpdate(codestring, oldDocstring)
    return new Promise((resolve, reject) => {
        const process = spawn(pyCommand[0], pyCommand.slice(1));

        let stdoutData = '';
        let stderrData = '';

        process.stdout.on('data', (data) => {
            stdoutData += data.toString();
        });

        process.stderr.on('data', (data) => {
            stderrData += data.toString();
        });

        process.on('close', (code) => {
            if (code !== 0) {
                return reject(new Error(`Process exited with code ${code}\n${stderrData}`));
            }

            try {
                console.info('Received data from python script');
                const result = JSON.parse(stdoutData);
                const formatted_result = result.new_docstring;
                resolve(formatted_result);
            } catch (parseError) {
                reject(parseError);
            }
        });

        process.on('error', (error) => {
            reject(error);
        });

        console.info('Sending data to python script');
        process.stdin.write(JSON.stringify({ codestring: codestring, old_docstring: oldDocstring }));
        process.stdin.end();
    });
}

function registerCodeActionsProvider() {
    codeActionProvider?.dispose();
    codeActionProvider = vscode.languages.registerCodeActionsProvider('python', {
        provideCodeActions
    }, { providedCodeActionKinds: [vscode.CodeActionKind.Refactor, vscode.CodeActionKind.RefactorRewrite] });
}

function provideCodeActions(document, range) {
    return lightbulbLines.flatMap(({ lineNumber, docstring_start, docstring_end, start_line, end_line }) => {
        if (executedActions.has(lineNumber)) return [];
        if (range.start.line !== lineNumber || range.end.line !== lineNumber) return [];
        
        const updateAction = createCodeAction('Update Docstring', LLAMADOC_UPDATE_DOCSTRING_COMMAND, vscode.CodeActionKind.RefactorRewrite, [lineNumber, docstring_start, docstring_end, start_line, end_line]);
        const dismissAction = createCodeAction('Dismiss', LLAMADOC_DISMISS_COMMAND, vscode.CodeActionKind.Refactor, [lineNumber]);

        updateAction.isPreferred = true;
        dismissAction.isPreferred = true;

        return [updateAction, dismissAction];
    });
}

function createCodeAction(title, command, kind, args = []) {
    return {
        title,
        command: { command, title, arguments: args },
        kind
    };
}

function spawnButtons(jsonObject, currentLineNumber, lineDiff = 0) {
    clearDecorations();
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    jsonObject.forEach(({ has_docstring, up_to_date, start_line, end_line, docstring_start_line, docstring_end_line }) => {
        if (!has_docstring || up_to_date) return;
        if (start_line < currentLineNumber) lineDiff = 0; // don't need linediff if action was before change in lines
        if (executedActions.has((start_line - lineDiff) - 1)) return;

        const lineNumber = start_line - 1;
        const decorationType = vscode.window.createTextEditorDecorationType({
            after: {
                contentText: "Expired Docstring!",
                color: '#f0f0f0',
                backgroundColor: '#333333',
                margin: '0 0 0 2em',
                border: '1px solid #444444',
                fontSize: '12px'
            }
        });
        editor.setDecorations(decorationType, [{ range: new vscode.Range(new vscode.Position(lineNumber, 75), new vscode.Position(lineNumber, 75)) }]);
        spawnLightbulbs(lineNumber, docstring_start_line, docstring_end_line, start_line, end_line);

        decorationTypes.push({ decorationType, lineNumber });
    });
}

function spawnLightbulbs(lineNumber, docstring_start, docstring_end, start_line, end_line) {
    lightbulbLines.push({ lineNumber, docstring_start, docstring_end, start_line, end_line });
    registerCodeActionsProvider();
}

function clearDecorations() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    decorationTypes.forEach(({ decorationType }) => {
        editor.setDecorations(decorationType, []);
        decorationType.dispose();
    });
    decorationTypes = [];
}

function clearSpecificDecoration(lineNumber) {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

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

function updateLineDifference(lineNumber, lineDiff) {
    jsonObject.forEach(docstring => {
        if (lineNumber < docstring.start_line) {
            docstring.start_line += lineDiff;
            docstring.end_line += lineDiff;
            docstring.docstring_start_line += lineDiff;
            docstring.docstring_end_line += lineDiff;
        } else if (lineNumber === docstring.start_line) {
            docstring.end_line += lineDiff;
            docstring.docstring_end_line += lineDiff;
            docstring.up_to_date = true;
        }
    });
    const editor = vscode.window.activeTextEditor;
    editor?.document.save();

    dismissAllExpiredDocstrings(true);
    spawnButtons(jsonObject, lineNumber, lineDiff);
    registerCodeActionsProvider();
}

function dismissAllExpiredDocstrings(hidden = false) {
    clearDecorations();
    lightbulbLines = [];
    registerCodeActionsProvider();
    if (!hidden) {
        vscode.window.showInformationMessage('Dismissed all expired docstrings.');
    }
}

module.exports = { activate, deactivate };
