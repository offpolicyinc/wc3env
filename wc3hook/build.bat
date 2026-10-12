@echo off
setlocal
if /i "%VSCMD_ARG_TGT_ARCH%"=="x86" goto compile
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo Install Visual Studio Build Tools with Desktop development with C++.
  exit /b 1
)
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_DIR=%%i"
if not defined VS_DIR exit /b 1
call "%VS_DIR%\VC\Auxiliary\Build\vcvars32.bat" >nul || exit /b 1
:compile
cd /d "%~dp0"
if not exist out mkdir out
set "JSON_FLAGS=/DYYJSON_DISABLE_WRITER=1 /DYYJSON_DISABLE_UTILS=1 /DYYJSON_DISABLE_FILE=1 /DYYJSON_DISABLE_INCR_READER=1 /DYYJSON_DISABLE_NON_STANDARD=1 /DYYJSON_READER_DEPTH_LIMIT=64"
cl /nologo /O2 /W3 /WX %JSON_FLAGS% /LD /I minhook\include main.c allocator.c transport.c clock.c step.c observe.c difficulty.c overlay.c chat.c metadata.c events.c act.c placement.c players.c setup.c instance.c diag.c rpc.c stage.c features.c masks.c yyjson\yyjson.c minhook\src\buffer.c minhook\src\hook.c minhook\src\trampoline.c minhook\src\hde\hde32.c /Fo:out\ /Fe:out\wc3hook.dll /link ole32.lib user32.lib winmm.lib || exit /b 1
cl /nologo /O2 /W3 /WX %JSON_FLAGS% ..\tests\native\test_core.c yyjson\yyjson.c /Fo:out\ /Fe:out\test_core.exe || exit /b 1
out\test_core.exe || exit /b 1
if not exist ..\src\wc3env\native mkdir ..\src\wc3env\native
copy /y out\wc3hook.dll ..\src\wc3env\native\wc3hook.dll >nul || exit /b 1
echo Built and tested wc3hook.dll; copied into the Python package.
