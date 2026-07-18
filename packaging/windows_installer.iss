; Inno Setup script for Painting Assist (Windows installer).
;
; Compiled in CI with:  iscc packaging\windows_installer.iss
; Expects the PyInstaller onedir output at dist\Painting Assist\ (which
; contains PaintingAssist.exe and its dependencies).

#define AppName "Painting Assist"
#define AppVersion "0.5.0"
#define AppPublisher "Vinay Williams"
#define AppExeName "PaintingAssist.exe"
#define AppURL "https://github.com/v-i-n-a-y/painting-assist"

[Setup]
; Paths below are resolved relative to SourceDir (the repo root), not this
; file's directory — iscc is invoked as `iscc packaging\windows_installer.iss`
; from the repo root, but without SourceDir it would resolve against packaging\.
SourceDir=..
AppId={{9C4B2A7E-1D3F-4E6A-8B2C-5A1F0E7D6C43}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=dist
OutputBaseFilename=PaintingAssist-{#AppVersion}-windows-x64-setup
SetupIconFile=painting_assist\resources\icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Recursively bundle the entire PyInstaller onedir output.
Source: "dist\Painting Assist\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
