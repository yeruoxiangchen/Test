/* Actual Chromium screenshots of pinned CodeMirror/xterm widgets.
 * Input is inert text, never executed by the browser. All task execution is
 * separately captured in Bubblewrap. This is a controlled tool capture.
 */
const fs = require('fs');
const path = require('path');
const { chromium } = require('/grader/node_modules/playwright');

(async () => {
  console.log('Loading capture plan');
  const plan = JSON.parse(fs.readFileSync('/grader/plan.json', 'utf8'));
  const version = fs.readdirSync('/grader/browsers').find(x => x.startsWith('chromium_headless_shell-'));
  const browser = await chromium.launch({
    executablePath: `/grader/browsers/${version}/chrome-linux/headless_shell`,
    headless: true,
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--disable-software-rasterizer'],
  });
  console.log('Chromium ready');
  try {
    const page = await browser.newPage({ viewport: {width: 1600, height: 1000}, deviceScaleFactor: 1 });
    await page.setContent('<html><head></head><body><div id="filename"></div><div id="pane"></div></body></html>');
    await page.addStyleTag({path:'/grader/node_modules/codemirror/lib/codemirror.css'});
    await page.addStyleTag({path:'/grader/node_modules/@xterm/xterm/css/xterm.css'});
    await page.addScriptTag({path:'/grader/node_modules/codemirror/lib/codemirror.js'});
    await page.addScriptTag({path:'/grader/node_modules/codemirror/mode/python/python.js'});
    await page.addScriptTag({path:'/grader/node_modules/@xterm/xterm/lib/xterm.js'});
    console.log('Widgets loaded');
    if (plan.some(item => /[^\x00-\x7f]/.test(item.text+item.filename)))
      throw new Error('This capture profile supports ASCII; choose an audited font for other glyphs');
    await page.addStyleTag({content:`
      * {box-sizing:border-box} body {margin:0;background:#252b35;color:#e8edf2}
      #filename {position:absolute;left:80px;top:180px;font:18px "DejaVu Sans Mono",monospace}
      #pane {position:absolute;left:70px;top:230px;width:1460px;height:650px;background:#fff;overflow:hidden}
      .CodeMirror {width:1460px;height:650px;font-family:"DejaVu Sans Mono",monospace;line-height:24px}
      .CodeMirror-gutters {display:none}
      .CodeMirror-cursor {display:none!important}
    `});
    for (const item of plan) {
      console.log(`Rendering ${item.id}`);
      const result = await page.evaluate(async item => {
        const pane = document.getElementById('pane');
        pane.innerHTML = '';
        const header = document.getElementById('filename');
        header.textContent = item.filename;
        const layout = [];
        const titleChars = [];
        for (let i=0; i<header.textContent.length; i++) {
          const r=document.createRange();r.setStart(header.firstChild,i);r.setEnd(header.firstChild,i+1);
          const b=r.getBoundingClientRect();
          titleChars.push({text:header.textContent[i],bbox:[Math.floor(b.x),Math.floor(b.y),Math.ceil(b.right),Math.ceil(b.bottom)]});
        }
        layout.push(titleChars);layout.push([]);
        let visible = '';
        if (item.kind === 'code') {
          const editor = CodeMirror(pane, {value:item.text,mode:'python',readOnly:'nocursor',lineNumbers:false,
            lineWrapping:false,viewportMargin:Infinity});
          editor.getWrapperElement().style.fontSize = item.font_size+'px';
          editor.refresh();
          visible = editor.getValue();
          for (let y=0; y<editor.lineCount(); y++) {
            const line=editor.getLine(y), row=[];
            for (let x=0; x<line.length; x++) {
              const a=editor.charCoords({line:y,ch:x},'page');
              const b=editor.charCoords({line:y,ch:x+1},'page');
              if (a.top>=230 && a.bottom<=880 && a.left>=70 && b.left<=1530)
                row.push({text:line[x],bbox:[Math.floor(a.left),Math.floor(a.top),Math.ceil(b.left),Math.ceil(a.bottom)]});
            }
            layout.push(row);
          }
        } else {
          const terminal = new Terminal({cols:130,rows:26,fontFamily:'"DejaVu Sans Mono",monospace',fontSize:item.font_size,
            lineHeight:1.15,convertEol:true,scrollback:500,theme:{background:'#17202a',foreground:'#e8edf2'}});
          terminal.open(pane);
          await new Promise(resolve=>terminal.write(item.text,resolve));
          await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
          const screen=terminal.element.querySelector('.xterm-screen').getBoundingClientRect();
          const cw=screen.width/terminal.cols,ch=screen.height/terminal.rows;
          const lines=[];
          for (let y=0;y<terminal.rows;y++) {
            const line=terminal.buffer.active.getLine(terminal.buffer.active.viewportY+y);
            const text=line ? line.translateToString(true) : '';
            const row=[];
            for (let x=0;x<text.length;x++) row.push({text:text[x],bbox:[Math.floor(screen.left+x*cw),
              Math.floor(screen.top+y*ch),Math.ceil(screen.left+(x+1)*cw),Math.ceil(screen.top+(y+1)*ch)]});
            layout.push(row);lines.push(text);
          }
          visible=lines.join('\n').trimEnd();
        }
        await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
        return {reference:item.filename+'\n\n'+visible,layout,
          crop_bbox:[70,225,1530,885],kind:item.kind,font_size:item.font_size,
          provenance:'controlled_real_widget_capture_not_user_submitted'};
      }, item);
      const directory = path.join('/workspace', item.id);
      fs.mkdirSync(directory,{recursive:true});
      await page.screenshot({path:path.join(directory,'screenshot.png')});
      fs.writeFileSync(path.join(directory,'screenshot.txt'), result.reference);
      fs.writeFileSync(path.join(directory,'screenshot.layout.json'), JSON.stringify(result.layout));
      fs.writeFileSync(path.join(directory,'capture.json'),JSON.stringify(result,null,2));
      console.log(`Captured ${item.id} / ${item.kind}`);
    }
    fs.writeFileSync('/workspace/browser.json',JSON.stringify({version:browser.version(),captures:plan.length,
      font:'DejaVu Sans Mono',glyph_scope:'ASCII',hardware_gpu:false,network:'unshared',organic_user_screenshots:0},null,2));
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
