const path = require('path')
const vscode = require('vscode');
const os = require('os')
/**
 * @param {vscode.ExtensionContext} context
 */

vscode.commands.registerCommand('llamadoc.myCommand', () => {
    vscode.window.showInformationMessage('My Command was invoked!');
});

function getUpdatedText(linenumber) {
	return "    \"\"\"This is the updated docstring\"\"\"\n"
}

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
	//let decorations = [];
	for (const pythonFunc of jsonObject) {
		if (!pythonFunc.has_docstring || pythonFunc.up_to_date) continue;
		const lineNumber = pythonFunc.start_line - 1;

		let buttonDecorationType = vscode.window.createTextEditorDecorationType({
			after: {
				textDecoration: 'none',
				color: '#ffffff',
				backgroundColor: '#007acc',
				fontWeight: 'bold',
				padding: '4px 8px',
			}
		});

		// Get the active text editor
		let editor = vscode.window.activeTextEditor;
		if (editor) {
			// Get the line where you want to add the button
			let startPosition = new vscode.Position(lineNumber, 75);
			let endPosition = new vscode.Position(lineNumber, 75);

			// Define the decoration
			let decoration = {
				range: new vscode.Range(startPosition, endPosition),
				renderOptions: {
					after: {
						contentText: "Found Out of Date Docstring!",
						margin: '0 0 0 4em',
					}
				}
			};
			let docstring_start = pythonFunc.docstring_start_line
			let docstring_end = pythonFunc.docstring_end_line
			let replacementText = getUpdatedText(lineNumber)
            
            spawnLightbulbs(lineNumber, docstring_start, docstring_end, replacementText)

			// Apply the decoration to the active text editor
			editor.setDecorations(buttonDecorationType, [decoration]);
		}
	}
}

function activate(context) {

	let button = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
	button.text = "$(megaphone) Check for Out of Date Docstrings";
	button.command = "llamadoc.scanFile";
	button.show();

	let disposable = vscode.commands.registerCommand('llamadoc.scanFile', function () {
		
		const { exec } = require('child_process');
		const argument = vscode.window.activeTextEditor.document.fileName;

		let windowsSystem = `python -u ${__dirname}\\find_docstrings.py "${argument}"`;
		let otherSystem = `python3 -u ${__dirname}/find_docstrings.py "${argument}"`;

		let pyCommand
		
		const platform = os.platform();
		
		if (platform == 'win32') {
			pyCommand = windowsSystem
		} else {
			pyCommand = otherSystem
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
