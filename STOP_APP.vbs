
Option Explicit

Dim shell, command, result

Set shell = CreateObject("WScript.Shell")

command = "powershell.exe -NoProfile -WindowStyle Hidden -Command " & _
          Chr(34) & _
          "$connections = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue; " & _
          "foreach ($connection in $connections) { " & _
          "$processId = $connection.OwningProcess; " & _
          "$process = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $processId); " & _
          "if ($process -and $process.CommandLine -match 'streamlit' -and " & _
          "$process.CommandLine -match 'web_app.py') { " & _
          "Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue " & _
          "} " & _
          "}" & _
          Chr(34)

result = shell.Run(command, 0, True)

If result = 0 Then
    MsgBox "Stop request completed.", vbInformation, "Local RAG Search"
Else
    MsgBox "Could not complete the stop request.", vbExclamation, "Local RAG Search"
End If
