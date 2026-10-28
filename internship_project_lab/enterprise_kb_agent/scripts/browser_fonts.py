"""Project-scoped CJK font for headless browser validation; no system mutation."""
import os
from pathlib import Path
from urllib.request import urlopen
from xml.sax.saxutils import escape

root=Path(os.environ['KB_DATA_DIR'])
fonts=root/'cache/fonts'
fonts.mkdir(parents=True,exist_ok=True)
url='https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf'
target=fonts/'NotoSansCJKsc-Regular.otf'
if not target.exists():
    temporary=fonts/'font.download'
    with urlopen(url,timeout=120) as response, temporary.open('wb') as output:
        while data:=response.read(1024*1024):
            output.write(data)
    if temporary.stat().st_size < 100000:
        raise RuntimeError('Font download is incomplete')
    temporary.replace(target)
if not (fonts/'LICENSE').exists():
    with urlopen('https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/LICENSE',timeout=30) as response:
        (fonts/'LICENSE').write_bytes(response.read())
config=fonts/'fonts.conf'
config.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig>'
                  '<include ignore_missing="yes">/etc/fonts/fonts.conf</include>'
                  '<dir>'+escape(str(fonts))+'</dir>'
                  '<cachedir>'+escape(str(root/'cache/fontconfig'))+'</cachedir></fontconfig>')
print('Project-scoped browser font configured:', config)
