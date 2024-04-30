const path = require('path')
const vscode = require('vscode');
const os = require('os')
/**
 * @param {vscode.ExtensionContext} context
 */


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
