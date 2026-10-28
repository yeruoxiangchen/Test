const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const Basis = require('./basis/basis_transcoder.js');

const table = Array.from({length:256}, (_, n) => {
  for(let k=0;k<8;k++) n = (n&1) ? (0xedb88320^(n>>>1)) : (n>>>1);
  return n>>>0;
});
function crc(data) { let c=0xffffffff; for(const b of data)c=table[(c^b)&255]^(c>>>8); return (c^0xffffffff)>>>0; }
function chunk(type, data) {
  const t=Buffer.from(type), len=Buffer.alloc(4), sum=Buffer.alloc(4);
  len.writeUInt32BE(data.length);sum.writeUInt32BE(crc(Buffer.concat([t,data])));
  return Buffer.concat([len,t,data,sum]);
}
function png(width,height,rgba) {
  const ihdr=Buffer.alloc(13);ihdr.writeUInt32BE(width,0);ihdr.writeUInt32BE(height,4);ihdr[8]=8;ihdr[9]=6;
  const scan=Buffer.alloc((width*4+1)*height);
  const src=Buffer.from(rgba);
  for(let y=0;y<height;y++) src.copy(scan,y*(width*4+1)+1,y*width*4,(y+1)*width*4);
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',ihdr),chunk('IDAT',zlib.deflateSync(scan)),chunk('IEND',Buffer.alloc(0))]);
}
(async()=>{
  const module=await Basis({wasmBinary:fs.readFileSync(path.join(__dirname,'basis/basis_transcoder.wasm'))});
  module.initializeBasis();
  const entries=JSON.parse(fs.readFileSync(process.argv[2]));
  let count=0;
  for(const item of entries){
    if(!fs.existsSync(item.output)){
      const file=new module.KTX2File(new Uint8Array(fs.readFileSync(item.input)));
      if(!file.isValid() || !file.startTranscoding())throw new Error('KTX2 decode failed: '+item.input);
      const w=file.getWidth(),h=file.getHeight();
      const rgba=new Uint8Array(file.getImageTranscodedSizeInBytes(0,0,0,13));
      if(!file.transcodeImage(rgba,0,0,0,13,0,-1,-1))throw new Error('RGBA transcode failed');
      if(rgba.length!==w*h*4)throw new Error('Unexpected RGBA shape');
      fs.writeFileSync(item.output,png(w,h,rgba));file.close();file.delete();
    }
    count++;
    if(count%25===0 || count===entries.length)console.log(JSON.stringify({decoded_textures:count,total:entries.length}));
  }
})().catch(e=>{console.error(e);process.exit(1)});
