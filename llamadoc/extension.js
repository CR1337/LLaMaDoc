const path = require('path')
const vscode = require('vscode');
const os = require('os')
/**
 * @param {vscode.ExtensionContext} context
 */


function spawnButtons(jsonObject) {
	let decorations = [];
	for (const func of jsonObject) {
		if (!func.has_docstring) continue;
		const lineNumber = func.start_line - 1;
	
		let buttonDecorationType = vscode.window.createTextEditorDecorationType({
			after: {
				margin: '0 0 0 4em',
				contentText: "$(sync) Update", // Example icon with text
				textDecoration: 'none', // Example property
				color: '#ffffff', // Example color
				backgroundColor: '#007acc', // Example background color
				fontWeight: 'bold', // Example font weight
				padding: '4px 8px', // Example padding
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
						contentText: "$(sync) Update",
						margin: '0 0 0 4em',
					}
				}
			};

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

// This method is called when your extension is deactivated
function deactivate() {}

module.exports = {
	activate,
	deactivate
}
