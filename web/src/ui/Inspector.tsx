import React from 'react'

export default function Inspector({ selected }: { selected: any }) {
  if (!selected) {
    return <div><h3 style={{marginTop:0}}>Inspector</h3><div>No selection</div></div>
  }
  const props = (selected?.properties && selected.properties) || {}
  const entries = typeof props?.getPropertyNames === 'function'
    ? props.getPropertyNames().map((k: string) => [k, props.getValue(k)] as const)
    : Object.entries(props || {})

  return (
    <div>
      <h3 style={{marginTop:0}}>Inspector</h3>
      <div style={{fontSize:12, opacity:0.8, marginBottom:8}}>Picked entity properties:</div>
      <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:6}}>
        {entries.map(([k,v]: any) => (
          <React.Fragment key={k}>
            <div style={{opacity:0.8}}>{k}</div>
            <div style={{textAlign:'right'}}>{String(v)}</div>
          </React.Fragment>
        ))}
      </div>
    </div>
  )
}
