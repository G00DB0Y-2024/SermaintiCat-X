' SilentLauncher-CMD.vbs
' ---------------------------------------------------------------
' Purpose: Silently launch start-backend.cmd with NO console window.
'          This is the standard Windows pattern: a tiny VBS host
'          runs a CMD/Batch file via WshShell.Run with window
'          style = 0 (hidden).
'
'          Use this for:
'            * Windows Startup folder (auto-start on boot)
'            * Task Scheduler "run whether user is logged on or not"
'            * Any shortcut where the user should never see a window
'
'          For visible/debug launches, just double-click start-backend.cmd.
'
' Stop:    End the python.exe / cmd.exe process in Task Manager.
' ---------------------------------------------------------------

Option Explicit

Const CMD_PATH = "F:\MyProjects\PDFAI\pdf-backen\start-backend.cmd"

' 0 = HIDDEN window  (the magic flag)
' False = do NOT wait for cmd to finish -- VBS returns immediately,
'         which is essential during boot so the Startup folder
'         doesn't get blocked.
Dim shell : Set shell = CreateObject("WScript.Shell")
shell.Run """" & CMD_PATH & """", 0, False

' Best-effort log so you can verify it fired after a reboot.
On Error Resume Next
Dim fso, ts
Set fso = CreateObject("Scripting.FileSystemObject")
Set ts  = fso.OpenTextFile("F:\MyProjects\PDFAI\pdf-backen\silent-launch.log", 8, True)
If Err.Number = 0 Then
    ts.WriteLine "[" & Now() & "] silent launch (cmd): " & CMD_PATH
    ts.Close
End If
On Error Goto 0

WScript.Quit 0
