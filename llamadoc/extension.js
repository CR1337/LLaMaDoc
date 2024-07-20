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


/**
 * Activates the extension by registering commands and creating the status bar buttons.
 * @param {vscode.ExtensionContext} context - The context for the extension.
 */
function activate(context) {
    registerCommands(context);
    displayStatusBarButtons(context);
}


/**
 * Deactivates the extension by disposing of the code action provider.
 */
function deactivate() {
    codeActionProvider?.dispose();
}


/**
 * Shows a loading animation / message in the status bar.
 * @param {string} message - The message to display.
 * @returns {vscode.Disposable} The status bar message.
 */
function showLoadingMessage(message) {
    const loadingMessage = vscode.window.setStatusBarMessage(`$(sync~spin) ${message}`);
    return loadingMessage;
}


/**
 * Registers the extension commands.
 * @param {vscode.ExtensionContext} context - The context for the extension.
 */
function registerCommands(context) {
    context.subscriptions.push(
        vscode.commands.registerCommand(LLAMADOC_UPDATE_DOCSTRING_COMMAND, updateDocstring),
        vscode.commands.registerCommand(LLAMADOC_DISMISS_COMMAND, dismissDocstring),
        vscode.commands.registerCommand(LLAMADOC_SCAN_FILE, findDocstrings),
        vscode.commands.registerCommand('llamadoc.dismissAll', dismissAllExpiredDocstrings)
    );
}


/**
 * Creates and displays the status bar buttons for the extension.
 * @param {vscode.ExtensionContext} context - The context for the extension.
 */
function displayStatusBarButtons(context) {
    const searchButton = createStatusBarButton("$(search) Find Outdated Docstrings", LLAMADOC_SCAN_FILE, vscode.StatusBarAlignment.Left, 100);
    context.subscriptions.push(searchButton);

    dismissAllButton = createStatusBarButton("$(close) Dismiss All Docstring Suggestions", 'llamadoc.dismissAll', vscode.StatusBarAlignment.Left, 99);
    context.subscriptions.push(dismissAllButton);
}


/**
 * Creates a status bar button.
 * @param {string} text - The text to display on the button.
 * @param {string} command - The command to execute when the button is pressed.
 * @param {vscode.StatusBarAlignment} alignment - The alignment of the button in the status bar.
 * @param {number} priority - The priority of the button.
 * @returns {vscode.StatusBarItem} The created status bar button.
 */
function createStatusBarButton(text, command, alignment, priority) {
    const button = vscode.window.createStatusBarItem(alignment, priority);
    button.text = text;
    button.command = command;
    button.show();
    return button;
}


/**
 * Registers the code action provider for the extension.
 */
function registerCodeActionsProvider() {
    codeActionProvider?.dispose();
    codeActionProvider = vscode.languages.registerCodeActionsProvider('python', {
        provideCodeActions
    }, { providedCodeActionKinds: [vscode.CodeActionKind.Refactor, vscode.CodeActionKind.RefactorRewrite] });
}


/**
 * Provides code actions for the specified document and range.
 * @param {vscode.TextDocument} document - The document to provide code actions for.
 * @param {vscode.Range} range - The range to provide code actions for.
 * @returns {vscode.CodeAction[]} The code actions.
 */
function provideCodeActions(document, range) {
    return lightbulbLines.flatMap(({ lineNumber, docstring_start, docstring_end, start_line, end_line }) => {
        if (executedActions.has(lineNumber)) return [];
        if (range.start.line !== lineNumber || range.end.line !== lineNumber) return [];
        
        const updateAction = createCodeAction('Update Docstring', LLAMADOC_UPDATE_DOCSTRING_COMMAND, vscode.CodeActionKind.RefactorRewrite, [lineNumber, docstring_start, docstring_end, start_line, end_line]);
        const dismissAction = createCodeAction('Dismiss Docstring Suggestion', LLAMADOC_DISMISS_COMMAND, vscode.CodeActionKind.Refactor, [lineNumber]);

        updateAction.isPreferred = true;
        dismissAction.isPreferred = true;

        return [updateAction, dismissAction];
    });
}


/**
 * Creates a code action.
 * @param {string} title - The title of the code action.
 * @param {string} command - The command to execute for the code action.
 * @param {vscode.CodeActionKind} kind - The kind of the code action.
 * @param {Array} args - The arguments for the command.
 * @returns {vscode.CodeAction} The created code action.
 */
function createCodeAction(title, command, kind, args = []) {
    return {
        title,
        command: { command, title, arguments: args },
        kind
    };
}


/**
 * Finds and tags out-of-date docstrings in the current file by calling the Python script.
 */
function findDocstrings() {
    executedActions.clear();
    lightbulbLines = [];
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) {
        vscode.window.showErrorMessage('No active editor found.');
        return;
    }

    const loadingMessage = showLoadingMessage('Scanning for outdated docstrings...');
    const pyCommand = getPythonCommandFindDocstrings(activeEditor.document.fileName);
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
        spawnExpiredDocstringTags(jsonObject);
        dismissAllButton.show();
    });
}


/**
 * Constructs the python command to find docstrings using the Python script.
 * @param {string} fileName - The name of the file to scan.
 * @returns {string} The command to execute.
 */
function getPythonCommandFindDocstrings(fileName) {
    const platform = os.platform();
    const scriptPath = platform === 'win32' ? `${__dirname}\\find_docstrings.py` : `${__dirname}/find_docstrings.py`;
    const pythonCommand = platform === 'win32' ? 'python -u' : 'python3 -u';
    return `${pythonCommand} ${scriptPath} "${fileName}"`;
}


/**
 * Creates the expired-docstring tags and lightbulbs for all functions with outdated docstrings in the current editor.
 * @param {Object[]} jsonObject - The JSON object containing the docstrings, the code and the line numbers.
 * @param {number} [currentLineNumber=0] - The current line number.
 * @param {number} [lineDiff=0] - The difference in documentation length. Might be needed to move the tag when updating code.
 */
function spawnExpiredDocstringTags(jsonObject, currentLineNumber, lineDiff = 0) {
    clearDecorations();
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    jsonObject.forEach(({ has_docstring, up_to_date, start_line, end_line, docstring_start_line, docstring_end_line }) => {
        if (!has_docstring || up_to_date) return;
        if (start_line < currentLineNumber) lineDiff = 0; // don't need line-difference if action was before change in lines
        if (executedActions.has((start_line - lineDiff) - 1)) return;

        const lineNumber = start_line - 1;
        const decorationType = vscode.window.createTextEditorDecorationType({
            after: {
                contentText: "Outdated Docstring!",
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


/**
 * Spawns lightbulbs for the specified line.
 * @param {number} lineNumber - The line number of the function definition.
 * @param {number} docstring_start - The start line of the docstring.
 * @param {number} docstring_end - The end line of the docstring.
 * @param {number} start_line - The start line of the code.
 * @param {number} end_line - The end line of the code.
 */
function spawnLightbulbs(lineNumber, docstring_start, docstring_end, start_line, end_line) {
    lightbulbLines.push({ lineNumber, docstring_start, docstring_end, start_line, end_line });
    registerCodeActionsProvider();
}


/**
 * Dismisses the docstring tag and lightbulb for the specified line.
 * @param {number} lineNumber - The line number of the function definition.
 */
function dismissDocstring(lineNumber) {
    clearSpecificDecoration(lineNumber);
    executedActions.add(lineNumber);
    vscode.window.showInformationMessage('Dismissed docstring suggestion.');
}


/**
 * Clears the decoration (tag and lightbulb) for a specific line.
 * @param {number} lineNumber - The line number for which to clear decorations.
 */
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


/**
 * Dismisses all expired docstrings by removing their tag and lightbulb.
 * @param {boolean} [hidden=false] - Whether to hide the information message.
 */
function dismissAllExpiredDocstrings(hidden = false) {
    clearDecorations();
    lightbulbLines = [];
    registerCodeActionsProvider();
    if (!hidden) {
        vscode.window.showInformationMessage('Dismissed all docstring suggestions.');
    }
}


/**
 * Clears all decorations (tags and lightbulbs) from the editor.
 */
function clearDecorations() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    decorationTypes.forEach(({ decorationType }) => {
        editor.setDecorations(decorationType, []);
        decorationType.dispose();
    });
    decorationTypes = [];
}


/**
 * Updates the docstring for the specified line. First has to get the updated docstring text, then replaces the docstring in the active file.
 * If the new docstring is shorter or longer than the old one, it also updates tags and lightbulbs for functions below the current one.
 * @param {number} lineNumber - The line number of the function definition.
 * @param {number} docstringStartLine - The start line of the docstring.
 * @param {number} docstringEndLine - The end line of the docstring.
 * @param {number} start_line - The start line of the code.
 * @param {number} end_line - The end line of the code.
 */
async function updateDocstring(lineNumber, docstringStartLine, docstringEndLine, start_line, end_line) {
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) {
        vscode.window.showErrorMessage('No active editor found.');
        return;
    }

    const document = activeEditor.document;
    const oldDocstring = document.getText(new vscode.Range(new vscode.Position(docstringStartLine - 1, 0), new vscode.Position(docstringEndLine, 0)));
    const codeString = document.getText(new vscode.Range(new vscode.Position(start_line - 1, 0), new vscode.Position(end_line, 0)));
    
    const loadingMessage = showLoadingMessage('Updating docstring...');
    try {
        const newDocstring = await getUpdatedDocstring(codeString, oldDocstring);
        const newDocstringLength = newDocstring.split('\n').length - 1
        const oldDocstringLength = oldDocstring.split('\n').length - 1  
        const lineDiff = newDocstringLength - oldDocstringLength
    
        clearSpecificDecoration(lineNumber);
        executedActions.add(lineNumber);
        vscode.window.showInformationMessage('Updated docstring.');
    
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


/**
 * Gets the updated docstring by calling the Python script.
 * @param {string} codeString - The function code string.
 * @param {string} oldDocstring - The old docstring.
 * @returns {Promise<string>} The updated docstring.
 */
async function getUpdatedDocstring(codeString, oldDocstring) {
    const pyCommand = getPythonCommandUpdateDocstring(codeString, oldDocstring)
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

        process.stdin.write(JSON.stringify({ codestring: codeString, old_docstring: oldDocstring }));
        process.stdin.end();
    });
}


/**
 * Constructs the python command to update docstrings using the Python script.
 * @returns {string[]} The command to execute.
 */
function getPythonCommandUpdateDocstring() {
    const platform = os.platform();
    const scriptPath = platform === 'win32' ? `${__dirname}\\update_docstrings.py` : `${__dirname}/update_docstrings.py`;
    const pythonCommand = platform === 'win32' ? 'python' : 'python3';
    return [pythonCommand, '-u', scriptPath];
}


/**
 * Updates the line difference after a docstring update. Necessary when the old docstring and new docstring have a different length.
 * @param {number} lineNumber - The line number of the function definition.
 * @param {number} lineDiff - The difference of docstring lengths, measured in lines.
 */
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
    spawnExpiredDocstringTags(jsonObject, lineNumber, lineDiff);
    registerCodeActionsProvider();
}


module.exports = { activate, deactivate };
