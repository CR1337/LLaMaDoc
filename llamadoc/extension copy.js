const path = require('path')
const vscode = require('vscode');
const os = require('os')
/**
 * @param {vscode.ExtensionContext} context
 */


function spawnButtons(jsonObject) {
	const buttonAmount = jsonObject.length;
	const editor = vscode.window.activeTextEditor;
	if (!editor) { 
		vscode.window.showInformationMessage(`No editor found!`);
		return ; 
	}

	const decorations = [];
	const buttonDecorationType = vscode.window.createTextEditorDecorationType({
        after: {
            contentText: "Button",
            color: new vscode.ThemeColor('button.foreground'),
            fontWeight: 'bold',
            cursor: 'pointer',
            textDecoration: 'none; padding-left: 8px;',
            margin: '0px 2px',
        }
    });

	for (let i = 0; i < buttonAmount; i++) {
		const line = jsonObject[i].start_line;
		let startPosition = new vscode.Position(lineNumber, 75);
		let endPosition = new vscode.Position(lineNumber, 75);
		const decoration = {
			range: new vscode.Range(line + i, 75, line + i, 75),
			renderOptions: {
				before: {
					contentText: `Button ${i}`,
					color: new vscode.ThemeColor('button.foreground'),
					fontWeight: 'bold',
					cursor: 'pointer',
					textDecoration: 'none; padding-right: 8px;',
					margin: '0px 2px',
				}
			}
		};
		decorations.push(decoration);
		context.subscriptions.push(
			vscode.commands.registerCommand(`extension.button${i}Command`, () => {
				vscode.window.showInformationMessage(`Button ${i} clicked!`);
			})
		);
		vscode.window.showInformationMessage(`Pushed button ${i}`);
	}

	editor.setDecorations(buttonDecorationType, decorations);
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