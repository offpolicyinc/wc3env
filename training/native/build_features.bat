@echo off
rem The hook's feature code (wc3hook/features.c) as a 64-bit library for training/native/test_features.py.
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_DIR=%%i"
call "%VS_DIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
if not exist out mkdir out
cl /nologo /O2 /W3 /WX /DFEAT_DLL /LD ..\..\wc3hook\features.c ..\..\wc3hook\masks.c /Fo:out\ /Fe:out\features.dll || exit /b 1
echo built training\native\out\features.dll
