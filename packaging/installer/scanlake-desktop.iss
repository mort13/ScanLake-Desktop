; Inno Setup script: per-user installer for the ScanLake Desktop player build (no admin rights).
;   python packaging/build.py ort
;   iscc /DAppVersion=0.1.0 packaging\installer\scanlake-desktop.iss
; Output: dist\installer\ScanLake-Desktop-Setup-<version>.exe
; For the full build, add /DFlavor=full (installs from dist\full\ScanLake Desktop).

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef Flavor
  #define Flavor "ort"
#endif
#if Flavor == "full"
  #define SourceDir "..\..\dist\full\ScanLake Desktop"
  #define OutputName "ScanLake-Desktop-Full-Setup-" + AppVersion
#else
  #define SourceDir "..\..\dist\ScanLake Desktop"
  #define OutputName "ScanLake-Desktop-Setup-" + AppVersion
#endif

[Setup]
AppId={{6E3C2B8A-4F0E-4C55-9A57-5CA0D35E5A11}
AppName=ScanLake Desktop
AppVersion={#AppVersion}
AppPublisher=ScanLake
AppPublisherURL=https://scanlake.rocks
DefaultDirName={localappdata}\Programs\ScanLake Desktop
DefaultGroupName=ScanLake Desktop
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\..\dist\installer
OutputBaseFilename={#OutputName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName=ScanLake Desktop
CloseApplications=yes

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\ScanLake Desktop"; Filename: "{app}\ScanLake Desktop.exe"
Name: "{autodesktop}\ScanLake Desktop"; Filename: "{app}\ScanLake Desktop.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Run]
Filename: "{app}\ScanLake Desktop.exe"; Description: "Start ScanLake Desktop"; Flags: nowait postinstall skipifsilent

; Settings (%APPDATA%\ScanLake\Desktop) and captured data (%LOCALAPPDATA%\ScanLake\Desktop)
; are kept on uninstall, so scans that were not uploaded yet are not lost.
