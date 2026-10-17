
Option Explicit

Dim shell, fso, projectDir, pythonExe, appFile, command

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

projectDir = fso.GetParentFolderName(WScript.ScriptFullName)

pythonExe = projectDir & "\.venv\Scripts\pythonw.exe"
appFile = projectDir & "\src\web_app.py"

If Not fso.FileExists(pythonExe) Then
    MsgBox "Python environment not found. Install the project dependencies first.", vbCritical, "Local RAG Search"
    WScript.Quit
End If

command = """" & pythonExe & """ -m streamlit run """ & appFile & """ --server.headless true"

shell.CurrentDirectory = projectDir

' 0 hides the terminal window.
' False allows the launcher to finish while Streamlit runs.
shell.Run command, 0, False

' Open the web interface.
WScript.Sleep 4000
shell.Run "http://localhost:8501", 1, False
