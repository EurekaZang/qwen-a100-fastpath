import json,glob,sys
files=sorted(glob.glob(sys.argv[1]+'/*.json'))
prev=None
for f in files:
    r=json.load(open(f)); q=r.get('req',{})
    msgs=q.get('messages',[])
    roles=[m['role'] for m in msgs]
    sysblocks=q.get('system')
    nsys=len(sysblocks) if isinstance(sysblocks,list) else (1 if sysblocks else 0)
    syslen=sum(len(b.get('text','')) for b in sysblocks) if isinstance(sysblocks,list) else len(sysblocks or '')
    inline=[i for i,m in enumerate(msgs) if m['role']=='system']
    print(f"#{r['idx']} {r['path']} st={r['status']} ttfb={r['ttfb']:.2f}s dt={r['dt']:.2f}s model={q.get('model')} max_tokens={q.get('max_tokens')} stream={q.get('stream')} thinking={q.get('thinking')} effort={q.get('output_config')} ntools={len(q.get('tools',[]))} sys_blocks={nsys} sys_chars={syslen} nmsg={len(msgs)} inline_sys_at={inline}")
    print("   roles:", ''.join({'user':'U','assistant':'A','system':'S'}[x] for x in roles))
    for i in inline:
        c=msgs[i]['content']; t=c if isinstance(c,str) else ' '.join(b.get('text','') for b in c)
        print(f"   inline sys[{i}] ({len(t)} chars): {t[:150]!r}")
    if prev is not None:
        # compare message lists
        pm=prev.get('messages',[])
        same=0
        for a,b in zip(pm,msgs):
            if json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True): same+=1
            else: break
        print(f"   vs prev: identical leading msgs={same}/{len(pm)}; system same={json.dumps(prev.get('system'))==json.dumps(q.get('system'))}; tools same={json.dumps(prev.get('tools'))==json.dumps(q.get('tools'))}")
    prev=q
