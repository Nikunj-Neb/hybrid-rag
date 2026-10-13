'use client';
import { useEffect, useRef, useState } from 'react';
import { ArrowUp, BookOpen, Check, ChevronDown, FileText, Layers3, Loader2, Plus, Search, Sparkles, UploadCloud } from 'lucide-react';

type Source = {number:number; filename:string;page:number|null;excerpt:string};
type Message = {role:'user'|'assistant';content:string;sources?:Source[]};
type DocumentInfo = {document_id:string;filename:string;chunks:number};
const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home(){
 const [docs,setDocs]=useState<DocumentInfo[]>([]);
 const [messages,setMessages]=useState<Message[]>([]);
 const [text,setText]=useState('');
 const [mode,setMode]=useState('hybrid');
 const [busy,setBusy]=useState(false);
 const [uploading,setUploading]=useState(false);
 const [error,setError]=useState('');
 const fileRef=useRef<HTMLInputElement>(null);
 const bottom=useRef<HTMLDivElement>(null);
 async function refresh(){try{const r=await fetch(`${API}/documents`);if(r.ok)setDocs((await r.json()).documents);}catch{}}
 useEffect(()=>{refresh()},[]);
 useEffect(()=>{bottom.current?.scrollIntoView({behavior:'smooth'})},[messages,busy]);
 async function upload(files:FileList|null){if(!files?.length)return;setError('');setUploading(true);try{for(const file of Array.from(files)){const body=new FormData();body.append('file',file);const res=await fetch(`${API}/upload`,{method:'POST',body});if(!res.ok){const data=await res.json().catch(()=>({}));throw new Error(data.detail||`Upload failed (${res.status})`);}}await refresh();}catch(e){setError(e instanceof Error?e.message:'Upload failed');}finally{setUploading(false);if(fileRef.current)fileRef.current.value='';}}
 async function send(q?:string){const question=(q??text).trim();if(!question||busy)return;setText('');setError('');setMessages(old=>[...old,{role:'user',content:question}]);setBusy(true);try{const res=await fetch(`${API}/chat`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question,mode})});const data=await res.json();if(!res.ok)throw new Error(data.detail||'Could not get an answer');setMessages(old=>[...old,{role:'assistant',content:data.answer,sources:data.sources}]);}catch(e){setError(e instanceof Error?e.message:'Could not get an answer');}finally{setBusy(false);}}
 return <div className="app">
  <aside className="sidebar">
   <div className="brand"><div className="brand-icon"><Layers3 size={20}/></div><div>ATLAS <span> / LAB</span></div></div>
   <button className="new-chat" onClick={()=>{setMessages([]);setError('')}}><Plus size={17}/> New conversation <span>↗</span></button>
   <div className="side-section"><div className="side-title">KNOWLEDGE BASE <span>{docs.length}</span></div>
   <input ref={fileRef} style={{display:'none'}} type="file" accept=".pdf,.txt,.md" multiple onChange={e=>upload(e.target.files)}/>
   <button className="upload" disabled={uploading} onClick={()=>fileRef.current?.click()}>{uploading?<Loader2 className="spin" size={17}/>:<UploadCloud size={17}/>} {uploading?'Processing documents...':'Upload documents'}</button>
   <div className="documents">{docs.length?docs.map(d=><div className="document" key={d.document_id}><FileText size={16}/><div className="doc-info"><span title={d.filename}>{d.filename}</span><small>{d.chunks} indexed chunks</small></div><Check size={14} className="check"/></div>):<div className="empty-docs"><BookOpen size={22}/><div>No sources yet</div><small>Upload PDFs, TXT or Markdown files to start asking questions.</small></div>}</div>
   </div><div className="sidebar-bottom"><div className="status-dot"/> Gemini + Qdrant <span>connected when configured</span></div>
  </aside>
  <main className="main"><header><div className="breadcrumbs">Workspace <span>/</span> <strong>Research assistant</strong></div><div className="header-right"><div className="online-dot"/> Hybrid retrieval engine</div></header>
  <div className="conversation"><div className="conversation-inner">
  {!messages.length?<div className="hero"><div className="eyebrow"><Sparkles size={15}/> DOCUMENT INTELLIGENCE, REIMAGINED</div><h1>Knowledge, <em>connected.</em></h1><p>Ask complex questions across your documents. Atlas combines semantic understanding and precision keyword search for answers backed by your sources.</p><div className="feature-tags"><span><Layers3 size={15}/> Hybrid RAG</span><span><Search size={15}/> BM25 + Vectors</span><span><BookOpen size={15}/> Source citations</span></div><div className="suggestions"><div>TRY ASKING</div>{['Summarize the main ideas in my documents','Find where my notes discuss a specific concept','Compare two theories mentioned in my files'].map(q=><button key={q} onClick={()=>send(q)}>{q}<ArrowUp size={15}/></button>)}</div></div>:<div className="chat-stream">{messages.map((m,i)=><div className={'message '+m.role} key={i}><div className="avatar">{m.role==='assistant'?<Layers3 size={18}/>: 'Y'}</div><div className="message-body"><div className="message-label">{m.role==='assistant'?'ATLAS':'YOU'}</div><div className="message-text">{m.content}</div>{m.sources?.length?<div className="sources"><div className="source-label">SOURCES RETRIEVED</div>{m.sources.map(s=><details key={s.number}><summary><FileText size={14}/><span>[{s.number}] {s.filename}{s.page?` · page ${s.page}`:''}</span><ChevronDown size={14}/></summary><p>{s.excerpt}</p></details>)}</div>:null}</div></div>)}{busy&&<div className="thinking"><Loader2 size={17} className="spin"/> Searching your knowledge base and generating an answer...</div>}</div>}
  <div ref={bottom}/></div></div>
  <div className="composer-wrap"><div className="composer"><textarea value={text} onChange={e=>setText(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}}} placeholder="Ask anything about your knowledge base..." rows={2}/><div className="composer-footer"><div className="select-wrap"><Layers3 size={15}/><select aria-label="Retrieval mode" value={mode} onChange={e=>setMode(e.target.value)}><option value="hybrid">Hybrid search</option><option value="semantic">Semantic only</option><option value="keyword">Keyword only</option></select></div><div className="composer-actions"><span>Enter to send</span><button disabled={busy||!text.trim()} onClick={()=>send()} aria-label="Send message"><ArrowUp size={18}/></button></div></div></div>{error&&<div className="error">{error}</div>}<div className="footer-note">Atlas uses Gemini to answer from your uploaded sources. Verify important information.</div></div>
  </main></div>;
}
