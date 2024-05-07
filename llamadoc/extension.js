const vscode = require('vscode');

/**
 * This method is called when your extension is activated
 * Your extension is activated the very first time the command is executed
 */
function activate(context) {
    console.log('Congratulations, your extension "my-extension" is now active!');

    let disposable = vscode.commands.registerCommand('extension.generateRandomHints', () => {
        // Get active text editor
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor!');
            return;
        }

        const lineCount = editor.document.lineCount;
        const n = Math.floor(Math.random() * 3) + 1; // Random number of hints to generate

        // Array to store line numbers
        const lineNumbers = [];

        // Generate n random line numbers
        for (let i = 0; i < n; i++) {
            const lineNumber = Math.floor(Math.random() * lineCount) + 1;
            lineNumbers.push(lineNumber);
            vscode.window.showInformationMessage(`Line Number: ${lineNumber}`);
        }

        // Show code action hints
        lineNumbers.forEach(lineNumber => {
            const position = new vscode.Position(lineNumber - 1, 0);
            const range = new vscode.Range(position, position);
            const diagnostic = new vscode.Diagnostic(range, `Line Number: ${lineNumber}`, vscode.DiagnosticSeverity.Hint);
            const codeActions = [];
            const showLineNumberCommand = {
                title: 'Show Line Number',
                command: 'extension.showLineNumber',
                arguments: [lineNumber]
            };
            const codeAction = new vscode.CodeAction('Show Line Number', vscode.CodeActionKind.QuickFix);
            codeAction.diagnostics = [diagnostic];
            codeAction.isPreferred = true;
            codeAction.command = showLineNumberCommand;
            codeActions.push(codeAction);
            vscode.languages.registerCodeActionsProvider('*', {
                provideCodeActions(document, range) {
                    return codeActions;
                }
            });
        });

        vscode.window.showInformationMessage(`${n} code action hints generated!`);
    });

    context.subscriptions.push(disposable);

    // Register command to show line number
    let showLineNumberDisposable = vscode.commands.registerCommand('extension.showLineNumber', (lineNumber) => {
        vscode.window.showInformationMessage(`Line Number: ${lineNumber}`);
    });

    context.subscriptions.push(showLineNumberDisposable);
}

module.exports = {
    activate
};
