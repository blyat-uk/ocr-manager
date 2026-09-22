; OCR Manager's Windows installer (Inno Setup 6). Built by packaging/build.py:
;   ISCC /DAppVersion=1.0.0 /DSourceDir=<tree> /DOutputDir=<dist>
;        /DOutputBaseFilename=ocr-manager-v1.0.0-win /DIconFile=<app.ico> installer.iss
;
; A per-user install (no administrator rights) into
; %LOCALAPPDATA%\Programs\OCR Manager. The OCR engine the app downloads on
; first run lives apart from it, in %LOCALAPPDATA%\ocr-manager; uninstalling
; asks whether to remove that too.

#ifndef AppVersion
  #error AppVersion must be defined
#endif

[Setup]
AppId={{6F2B3C1E-8A4D-4E8B-9C1F-0B7D5E2A9C41}
AppName=OCR Manager
AppVersion={#AppVersion}
AppVerName=OCR Manager {#AppVersion}
AppPublisher=blyat-uk
AppPublisherURL=https://github.com/blyat-uk/ocr-manager
AppSupportURL=https://github.com/blyat-uk/ocr-manager/issues
DefaultDirName={localappdata}\Programs\OCR Manager
DefaultGroupName=OCR Manager
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OutputDir}
OutputBaseFilename={#OutputBaseFilename}
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\OCR Manager.exe
UninstallDisplayName=OCR Manager
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
WizardStyle=modern
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; An upgrade replaces the whole tree: files a new version dropped must not linger.
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\src"
Type: filesandordirs; Name: "{app}\bin"

[Icons]
Name: "{autoprograms}\OCR Manager"; Filename: "{app}\OCR Manager.exe"
Name: "{autodesktop}\OCR Manager"; Filename: "{app}\OCR Manager.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\OCR Manager.exe"; Description: "{cm:LaunchProgram,OCR Manager}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Engine: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    Engine := ExpandConstant('{localappdata}\ocr-manager');
    if DirExists(Engine) then
      if SuppressibleMsgBox('Also remove the downloaded OCR engine and logs?' + #13#10 + #13#10 + Engine,
                            mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES then
        DelTree(Engine, True, True, True);
  end;
end;
