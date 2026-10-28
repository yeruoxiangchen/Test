const $ = (q) => document.querySelector(q);
const escape = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let me, bases = [], conversationId = null, activeRun = null, feedbackRun = null, currentTab = 'chat', toastTimer;
const statuses = {queued:'等待入库',processing:'正在处理',ready:'已就绪',error:'处理失败',answered:'已完成',insufficient:'资料不足',clarify:'待补充信息',running:'执行中',cancelled:'已停止'};
function toast(message) { $('#toast').textContent = message; $('#toast').classList.remove('hidden'); clearTimeout(toastTimer); toastTimer = setTimeout(() => $('#toast').classList.add('hidden'), 4500); }
async function api(path, options = {}) {
  const headers = {'X-KB-Client':'web', ...(options.headers || {})};
  if (options.body && !(options.body instanceof FormData)) { headers['Content-Type'] = 'application/json'; options.body = JSON.stringify(options.body); }
  const response = await fetch('/api' + path, {...options, headers, credentials:'same-origin'});
  if (!response.ok) { const error = await response.json().catch(() => ({})); if (response.status === 401 && path !== '/login') showLogin(); throw new Error(typeof error.detail === 'string' ? error.detail : '请求失败，请检查输入后重试'); }
  return response.json();
}
function showLogin() { $('#workspace').classList.add('hidden'); $('#login-view').classList.remove('hidden'); me = null; }
function date(value) { return new Date(value * 1000).toLocaleString('zh-CN', {month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'}); }
async function enter() {
  me = await api('/me'); $('#login-view').classList.add('hidden'); $('#workspace').classList.remove('hidden');
  $('#username').textContent = me.username; $('#role-name').textContent = me.role === 'admin' ? '空间管理员' : '团队成员'; $('#avatar').textContent = me.username.slice(0,1).toUpperCase();
  document.querySelectorAll('.admin-only').forEach(el => el.classList.toggle('hidden', me.role !== 'admin'));
  $('#mode-badge').textContent = me.mode === 'agent' ? '模型问答' : '原文检索';
  $('#mode-hint').textContent = me.mode === 'agent' ? '根据已读取的文档生成回答' : '当前提供原文摘录，未启用模型生成';
  await Promise.all([refreshBases(), refreshHistory()]);
}
async function refreshBases() {
  bases = await api('/bases');
  for (const selector of ['#chat-base','#document-base']) {
    const previous = $(selector).value;
    $(selector).innerHTML = (selector === '#chat-base' ? '<option value="">全部可访问知识库</option>' : '') + bases.map(b => `<option value="${escape(b.id)}">${escape(b.name)}</option>`).join('');
    if ([...$(selector).options].some(o => o.value === previous)) $(selector).value = previous;
  }
  $('#base-cards').innerHTML = bases.map(b => `<article class="base-card"><h3>${escape(b.name)}</h3><p>${b.document_count} 份文档 · ${b.visibility === 'all' ? '全部成员' : '仅管理员'}</p><p>${escape(b.description)}</p></article>`).join('') || '<p class="muted">还没有知识库，请管理员创建一个。</p>';
}
async function refreshHistory() { const history = await api('/conversations'); $('#history').innerHTML = history.map(c => `<button data-conversation="${escape(c.id)}" title="${escape(c.title)}">${escape(c.title)}</button>`).join('') || '<div class="small muted">开始提问后会保存在这里</div>'; }
async function tab(name) {
  currentTab = name; for (const t of ['chat','documents','feedback']) $(`#${t}-view`).classList.toggle('hidden', t !== name);
  document.querySelectorAll('[data-tab]').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  $('#page-name').textContent = {chat:'知识问答',documents:'文档资料',feedback:'用户反馈'}[name];
  if (name === 'documents') { await refreshBases(); await refreshDocuments(); }
  if (name === 'feedback') await refreshFeedback();
}
const welcomeMarkup = $('#chat-messages').innerHTML;
function newChat() { conversationId = null; activeRun = null; $('#chat-messages').innerHTML = welcomeMarkup; setBusy(false); showEvidence({citations:[], events:[]}); tab('chat').catch(e=>toast(e.message)); $('#question').focus(); }
function setBusy(value) { $('#send').disabled = value; $('#cancel-run').classList.toggle('hidden', !value); }
function message(question, run) {
  const block = document.createElement('div'); block.dataset.runBlock = run.id;
  block.innerHTML = `<div class="message user">${escape(question)}</div><div class="message answer"><div class="message-label">✦ 知页 <span class="muted">· ${escape(statuses[run.status] || run.status)}</span></div><div class="answer-body"></div><div class="answer-actions"></div></div>`;
  fillMessage(block, run); $('#chat-messages').append(block); return block;
}
function fillMessage(block, run) {
  const body = block.querySelector('.answer-body'), actions = block.querySelector('.answer-actions');
  block.querySelector('.message-label').innerHTML = `✦ 知页 <span class="muted">· ${escape(statuses[run.status] || run.status)}</span>`;
  body.classList.toggle('working', ['queued','running'].includes(run.status));
  body.textContent = run.answer || run.error || (run.status === 'cancelled' ? '已停止本次任务。' : '正在检索资料，整理可用证据…');
  actions.innerHTML = `<button data-inspect="${escape(run.id)}">来源与过程 ↗</button>` + (run.answer ? `<button data-feedback="${escape(run.id)}">评价回答</button>` : '');
}
function showEvidence(run) {
  $('#sources').innerHTML = (run.citations || []).map(s => `<button class="source-card" data-document="${escape(s.document_id)}" data-chunk="${s.id}"><strong>[${escape(s.label)}] ${escape(s.filename)}</strong><small>${escape(s.base_name)}${s.page ? ' · 第 '+s.page+' 页' : ''}</small><p>${escape(s.text.slice(0,120))}…</p></button>`).join('') || '<div class="empty-small">尚未引用资料</div>';
  const names = {search:'检索资料',read_chunk:'读取原文',list_documents:'查看文档',finish:'提交答案'};
  $('#trace').innerHTML = (run.events || []).map(e => {
    let title = e.kind, detail = '';
    if (e.kind === 'started') { title = e.payload.resumed ? '恢复任务' : '开始任务'; detail = e.payload.mode === 'agent' ? '模型问答' : '原文检索'; }
    if (e.kind === 'tool') { title = names[e.payload.name] || e.payload.name; const r = e.payload.result; detail = r.error || (r.results ? `找到 ${r.results.length} 条候选片段` : r.filename || '已完成'); if (r.warnings?.length) detail += '；' + r.warnings.join('；'); }
    if (e.kind === 'model') { title = '模型处理'; detail = `第 ${e.payload.turn} 轮`; }
    if (e.kind === 'finished') { title = '完成'; detail = `引用 ${e.payload.citations} 条来源`; }
    if (e.kind === 'error') { title = '执行失败'; detail = e.payload.message; }
    if (e.kind === 'cancelled') title = '任务停止';
    return `<div class="trace-item"><b>${escape(title)}</b><small>${escape(detail)}</small></div>`;
  }).join('') || '<p class="muted small">暂无执行记录</p>';
}
async function monitor(runId, selectedConversation) {
  activeRun = runId; setBusy(true);
  try {
    while (conversationId === selectedConversation && activeRun === runId) {
      const run = await api('/runs/' + runId);
      if (conversationId !== selectedConversation || activeRun !== runId) break;
      const block = document.querySelector(`[data-run-block="${runId}"]`); if (block) fillMessage(block, run); showEvidence(run);
      if (!['queued','running'].includes(run.status)) { activeRun = null; setBusy(false); await refreshHistory(); break; }
      await new Promise(r=>setTimeout(r,700));
    }
  } catch (e) { if (activeRun === runId) { activeRun = null; setBusy(false); } toast(e.message); }
}
async function openConversation(id) {
  activeRun = null; setBusy(false); conversationId = id; const data = await api('/conversations/' + id); if (conversationId !== id) return;
  await tab('chat'); $('#chat-messages').innerHTML = ''; data.runs.forEach(run => message(run.question,run));
  const last = data.runs.at(-1); if (last) { showEvidence(await api('/runs/' + last.id)); if (['queued','running'].includes(last.status)) monitor(last.id,id); }
  $('#chat-messages').scrollTop = $('#chat-messages').scrollHeight;
}
async function refreshDocuments() {
  if (!me) return;
  const baseId = $('#document-base').value; const docs = await api('/documents' + (baseId ? '?base_id=' + encodeURIComponent(baseId) : ''));
  $('#documents').innerHTML = docs.map(d => `<tr><td><button class="text-button" data-document="${escape(d.id)}">▤ ${escape(d.filename)}</button>${d.error ? `<div class="error small">${escape(d.error)}</div>` : ''}</td><td><span class="status ${escape(d.status)}">${escape(statuses[d.status])}</span></td><td>${d.chunk_count}</td><td>${date(d.created_at)}</td><td>${me.role === 'admin' ? `<button class="text-button" data-reindex="${escape(d.id)}">重新索引</button><button class="text-button" data-delete="${escape(d.id)}">删除</button>` : '—'}</td></tr>`).join('') || '<tr><td colspan="5" class="muted">暂无文档。上传资料后，即可开始问答。</td></tr>';
}
async function source(id, chunk) {
  const data = await api('/documents/' + id); $('#source-title').textContent = data.filename; $('#source-download').href = '/api/documents/' + encodeURIComponent(id) + '/download';
  $('#source-content').innerHTML = data.chunks.map(c => `<div class="source-chunk" id="chunk-${c.id}"><small>片段 ${c.ordinal+1}${c.page ? ' · 第 '+c.page+' 页' : ''}</small><pre>${escape(c.text)}</pre></div>`).join('') || '<p class="muted">文档还未完成解析。</p>';
  $('#source-dialog').showModal(); if (chunk) document.getElementById('chunk-'+chunk)?.scrollIntoView({block:'start'});
}
async function refreshFeedback() { const rows = await api('/feedback'); $('#feedback-list').innerHTML = rows.map(f => `<article class="feedback-card"><small>${escape(f.username)} · ${date(f.created_at)} · ${f.rating === 1 ? '有帮助' : '需要改进'}</small><h3>${escape(f.question)}</h3><pre>${escape(f.answer)}</pre><p>${escape(f.comment || '未填写补充说明')}</p></article>`).join('') || '<p class="muted">还没有用户反馈。</p>'; }
$('#login-form').addEventListener('submit',async e=>{e.preventDefault(); const button=e.target.querySelector('button');button.disabled=true; $('#login-error').textContent=''; try{await api('/login',{method:'POST',body:Object.fromEntries(new FormData(e.target))});e.target.reset();await enter();}catch(err){$('#login-error').textContent=err.message;}finally{button.disabled=false;}});
$('#logout').addEventListener('click',async()=>{try{await api('/logout',{method:'POST'});activeRun=null;showLogin();}catch(e){toast(e.message);}});
$('#new-chat').addEventListener('click',newChat);
$('#question-form').addEventListener('submit',async e=>{e.preventDefault(); const question=$('#question').value.trim();if(!question||activeRun)return;setBusy(true);try{const result=await api('/chat',{method:'POST',body:{question,conversation_id:conversationId,base_ids:$('#chat-base').value?[$('#chat-base').value]:[]}});conversationId=result.conversation_id;$('#welcome')?.remove();message(question,{id:result.run_id,status:'queued'});$('#question').value='';$('#chat-messages').scrollTop=$('#chat-messages').scrollHeight;await refreshHistory();monitor(result.run_id,conversationId);}catch(err){setBusy(false);toast(err.message);}});
$('#question').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();$('#question-form').requestSubmit();}});
$('#cancel-run').addEventListener('click',async()=>{if(activeRun){try{await api('/runs/'+activeRun+'/cancel',{method:'POST'});toast('已请求停止，正在结束当前步骤');}catch(e){toast(e.message);}}});
$('#document-base').addEventListener('change',()=>refreshDocuments().catch(e=>toast(e.message)));
$('#upload').addEventListener('click',()=>{if(!$('#document-base').value){toast('请先创建知识库');return;}$('#file-input').click();});
$('#file-input').addEventListener('change',async e=>{const files=[...e.target.files];$('#upload').disabled=true;try{for(let i=0;i<files.length;i++){const form=new FormData();form.set('base_id',$('#document-base').value);form.set('file',files[i]);$('#upload-progress').textContent=`正在上传 ${i+1}/${files.length}：${files[i].name}`;const result=await api('/documents',{method:'POST',body:form});if(result.duplicate)toast(files[i].name+' 已存在，未重复导入');}$('#upload-progress').textContent='上传完成，后台正在解析文档。';await refreshBases();await refreshDocuments();}catch(err){$('#upload-progress').textContent=err.message;toast(err.message);}finally{$('#upload').disabled=false;e.target.value='';}});
$('#new-base').addEventListener('click',()=>$('#base-dialog').showModal());
$('#base-form').addEventListener('submit',async e=>{e.preventDefault();try{const result=await api('/bases',{method:'POST',body:Object.fromEntries(new FormData(e.target))});$('#base-dialog').close();e.target.reset();await refreshBases();$('#document-base').value=result.id;await refreshDocuments();toast('知识库已创建');}catch(err){toast(err.message);}});
$('#add-user').addEventListener('click',()=>$('#user-dialog').showModal());
$('#user-form').addEventListener('submit',async e=>{e.preventDefault();try{await api('/users',{method:'POST',body:Object.fromEntries(new FormData(e.target))});$('#user-dialog').close();e.target.reset();toast('成员账号已创建');}catch(err){toast(err.message);}});
$('#feedback-form').addEventListener('submit',async e=>{e.preventDefault();const body=Object.fromEntries(new FormData(e.target));body.rating=Number(body.rating);try{await api('/runs/'+feedbackRun+'/feedback',{method:'POST',body});$('#feedback-dialog').close();e.target.reset();toast('已收到反馈，谢谢');}catch(err){toast(err.message);}});
document.addEventListener('click',async e=>{const b=e.target.closest('button');if(!b)return;try{if(b.dataset.tab)await tab(b.dataset.tab);if(b.dataset.close)$('#'+b.dataset.close).close();if(b.dataset.conversation)await openConversation(b.dataset.conversation);if(b.closest('.suggestions')){$('#question').value=b.textContent;$('#question-form').requestSubmit();}if(b.dataset.document)await source(b.dataset.document,b.dataset.chunk);if(b.dataset.inspect){const run=await api('/runs/'+b.dataset.inspect);showEvidence(run);if(matchMedia('(max-width:800px)').matches){$('#source-title').textContent='来源与执行记录';$('#source-download').removeAttribute('href');$('#source-content').innerHTML=$('#sources').innerHTML+$('#trace').innerHTML;$('#source-dialog').showModal();}}if(b.dataset.feedback){feedbackRun=b.dataset.feedback;$('#feedback-dialog').showModal();}if(b.dataset.reindex){await api('/documents/'+b.dataset.reindex+'/reindex',{method:'POST'});await refreshDocuments();}if(b.dataset.delete&&confirm('删除该文档及其索引？历史回答保留，但原文将不可访问。')){await api('/documents/'+b.dataset.delete,{method:'DELETE'});await refreshBases();await refreshDocuments();}}catch(err){toast(err.message);}});
setInterval(()=>{if(me&&currentTab==='documents')refreshDocuments().catch(()=>{});},2500);
enter().catch(()=>showLogin());
