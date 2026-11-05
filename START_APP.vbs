Option Explicit
Dim shell, fso, root, python, app, command, versions, i, url
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)
versions = Array(".venv-lucy-314", ".venv-lucy", ".venv")
python = ""
For i = 0 To UBound(versions)
    If fso.FileExists(root & "\" & versions(i) & "\Scripts\pythonw.exe") Then
        python = root & "\" & versions(i) & "\Scripts\pythonw.exe"
        Exit For
    End If
Next
If python = "" Then
    MsgBox "LUCY is not installed. First run setup-windows.ps1.", vbExclamation, "LUCY"
    WScript.Quit 1
End If
app = root & "\src\web_app.py"
If Not fso.FileExists(app) Then
    MsgBox "Missing src\web_app.py", vbCritical, "LUCY"
    WScript.Quit 1
End If
shell.CurrentDirectory = root
' Start Ollama if installed but its local API is not running.
On Error Resume Next
Dim http, alive
Set http = CreateObject("MSXML2.XMLHTTP")
http.open "GET", "http://127.0.0.1:11434/api/tags", False
http.send
alive = (Err.Number = 0 And http.status = 200)
Err.Clear
On Error GoTo 0
If Not alive Then
    Dim ollama
    ollama = shell.ExpandEnvironmentStrings("%LOCALAPPDATA%") & "\Programs\Ollama\ollama.exe"
    If Not fso.FileExists(ollama) Then
        ollama = shell.ExpandEnvironmentStrings("%LOCALAPPDATA%") & "\Ollama\ollama.exe"
    End If
    If fso.FileExists(ollama) Then
        shell.Run Chr(34) & ollama & Chr(34) & " serve", 0, False
        WScript.Sleep 4000
    End If
End If
' Check whether Streamlit is already running to avoid duplicate launches.
On Error Resume Next
Set http = CreateObject("MSXML2.XMLHTTP")
http.open "GET", "http://127.0.0.1:8501/_stcore/health", False
http.send
alive = (Err.Number = 0 And http.status = 200)
Err.Clear
On Error GoTo 0
If Not alive Then
    command = Chr(34) & python & Chr(34) & " -m streamlit run " & Chr(34) & app & Chr(34) & " --server.headless true --server.address 127.0.0.1"
    shell.Run command, 0, False
    WScript.Sleep 4000
End If
shell.Run "http://localhost:8501", 1, False
