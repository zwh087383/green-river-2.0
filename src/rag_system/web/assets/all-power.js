// Full-power controls are separate from the original arm-only controls.
(() => {
  const on=document.getElementById('allPowerOnBtn'), off=document.getElementById('allPowerOffBtn');
  const status=document.getElementById('allPowerStatus'), message=document.getElementById('allPowerMessage');
  if (!on || !off || !status || !message) return;
  let state=null,sending=false,reading=false,pending=null,revision=0;
  const idle=()=>['ready','powered_off'].includes(state?.state);
  const known=()=>Array.isArray(state?.trunk?.power) && state.trunk.power.length===6
    && [state.power?.left,state.power?.right,...state.trunk.power].every(v=>v===0 || v===1)
    && state.power_feedback_age_ms<=2000;
  const allOn=()=>known() && [state.power.left,state.power.right,...state.trunk.power].every(v=>v===0);
  const allOff=()=>known() && [state.power.left,state.power.right,...state.trunk.power].every(v=>v===1);
  const powerLabel=v=>v===0?'已上电':v===1?'已断电':'未知';
  function render() {
    on.disabled=sending || !!pending || !idle() || !known() || allOn();
    off.disabled=sending || !!pending || !idle() || !known() || allOff();
    if (!known()) status.textContent='双臂或躯干状态未确认';
    else {
      const values=state.trunk.power;
      const body=values.slice(0,4).every(v=>v===0)?'已上电':values.slice(0,4).every(v=>v===1)?'已断电':'部分上电';
      const neck=values.slice(4).every(v=>v===0)?'已上电':values.slice(4).every(v=>v===1)?'已断电':'部分上电';
      status.textContent=`左臂${powerLabel(state.power.left)} / 右臂${powerLabel(state.power.right)} · 腰腿${body} · 颈部${neck}`;
    }
    if (state?.state==='fault') message.textContent=`操作已停止：${state.detail || '请检查机器人'}。系统不会自动重试。`;
    else if (state?.state==='powering_on_all') message.textContent='全部上电并归位中，请等待位置反馈确认。';
    else if (state?.state==='powering_off_all') message.textContent='全部归位并断电中，请等待使能及抱闸反馈确认。';
  }
  function observe() {
    if (!pending || !state || !['ready','powered_off','fault'].includes(state.state)) return;
    if (state.request_id!==pending.token || state.purpose!==`all_power_${pending.power}_home`)
      message.textContent='操作结果未确认，请核对当前状态；不会自动重发。';
    else if (state.state==='fault') message.textContent=`全部电源操作失败：${state.detail || '请检查机器人'}。`;
    else if (pending.power==='on' && allOn() && state.at_preset && state.trunk.at_home)
      message.textContent='全部上电并归位完成：双臂预置位及身体初始姿态已确认。';
    else if (pending.power==='off' && allOff() && state.trunk.brakes?.slice(0,4).every(v=>v===0))
      message.textContent='全部归位并断电完成，双臂及躯干退出使能、腰腿抱闸已确认。';
    else message.textContent='操作返回，但部分位置或电源反馈未确认，请现场核对。';
    pending=null;
  }
  async function refresh() {
    if (reading || sending) return;
    reading=true;const version=revision;
    try {
      const response=await fetch('/api/robot/motion/state',{cache:'no-store',signal:AbortSignal.timeout(2000)});
      if (!response.ok) throw new Error('state');
      const data=await response.json();
      if (version!==revision) return;
      state=data;observe();
    } catch (_) {if(version===revision)state=null;}
    finally {reading=false;render();}
  }
  async function change(power) {
    if (sending || pending || (power==='on'?on.disabled:off.disabled)) return;
    sending=true;revision++;render();
    message.textContent=power==='on'?'正在提交全部上电并归位…':'正在提交全部归位并断电…';
    try {
      const requestId=globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
      const response=await fetch('/api/robot/motion/all-power',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({request_id:requestId,power}),signal:AbortSignal.timeout(2000)});
      const data=await response.json();
      if (!response.ok) message.textContent=data.status==='busy'?'机器人正在执行动作，请等待完成。':typeof data.detail==='string'?data.detail:'全部电源操作未被接受，请检查状态。';
      else if (['accepted','duplicate'].includes(data.status)) {
        state=data;pending={power,token:data.request_id};message.textContent='操作已接收，等待执行结果…';
      } else {state=data;message.textContent=data.status==='already_on'?'全部已上电。':'全部已断电。';}
    } catch (_) {state=null;message.textContent='未确认操作回执，请先检查机器人；系统不会自动重试。';}
    finally {sending=false;render();await refresh();}
  }
  on.addEventListener('click',()=>change('on'));off.addEventListener('click',()=>change('off'));
  void refresh();setInterval(()=>{if(!document.hidden)void refresh();},1000);
})();
