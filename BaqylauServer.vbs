Dim sh, dir
dir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
Set sh = CreateObject("Wscript.Shell")
sh.CurrentDirectory = dir
sh.Run "pythonw -m server.app", 0, False
