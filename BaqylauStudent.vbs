Dim sh, dir, fso
dir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
Set sh = CreateObject("Wscript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
sh.CurrentDirectory = dir
If fso.FileExists(dir & "BaqylauStudent.exe") Then
  sh.Run """" & dir & "BaqylauStudent.exe""", 1, False
Else
  sh.Run "pythonw """ & dir & "main_gui.py""", 0, False
End If
