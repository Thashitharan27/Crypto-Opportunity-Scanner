Option Explicit

Dim shell, files, root, pythonw, appPath
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")

root = files.GetParentFolderName(WScript.ScriptFullName)
pythonw = files.BuildPath(root, ".venv\Scripts\pythonw.exe")
appPath = files.BuildPath(root, "app.py")

If Not files.FileExists(pythonw) Then
    MsgBox "The Python environment was not found:" & vbCrLf & pythonw & vbCrLf & vbCrLf & _
           "Create .venv and install requirements before launching.", _
           vbCritical, "Crypto Strategy Lab"
    WScript.Quit 1
End If

shell.CurrentDirectory = root
shell.Run """" & pythonw & """ """ & appPath & """", 0, False
