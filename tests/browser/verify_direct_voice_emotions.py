"""Controlled PCM reply + delayed emotion regression; no cloud requests."""
from __future__ import annotations

import argparse
import asyncio
import base64
import json

from playwright.async_api import async_playwright


async def main(root: str) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
            args=["--autoplay-policy=no-user-gesture-required", "--disable-background-timer-throttling", "--disable-renderer-backgrounding"])
        context = await browser.new_context()
        await context.add_init_script("localStorage.setItem('rag.wakeupEnabled','0'); localStorage.setItem('rag.autoSpeak','1');")
        page = await context.new_page()
        display = await context.new_page()
        page.set_default_timeout(10000)
        display.set_default_timeout(10000)
        tts_requests, errors = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        await context.route("**/api/voice/tts**", lambda route: (tts_requests.append(route.request.url), route.abort())[1])

        async def reply(route):
            audio = route.request.post_data_json["audio"]
            closing = route.request.post_data_json["query"] == "结束语流程测试"
            text = "太好了，志愿者成功救助了动物！" if audio else "我有点疑惑，你说的是哪一条河呢？"
            events = [{"type": "meta", "retrieval": {"reason": "direct_retrieval"}}]
            if closing:
                events.extend([
                    {"type": "tool_result", "name": "end_conversation", "result": {"end_session": True}},
                    {"type": "meta", "retrieval": {"reason": "end_session"}},
                ])
            events.append({"type": "delta", "text": text})
            if audio:
                events.append({"type": "audio", "audio": base64.b64encode(b"\0\0" * 52800).decode(),
                    "sample_rate": 24000, "channels": 1, "format": "pcm16"})
            events.append({"type": "final", "result": {"answer": text}})
            await route.fulfill(status=200, content_type="application/x-ndjson",
                body="\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n")

        async def emotion(route):
            text = route.request.post_data_json["text"]
            await asyncio.sleep(0.8)
            try:
                await route.fulfill(json={"tone": "curious" if "疑惑" in text else "happy", "confidence": 0.95, "source": "test"})
            except Exception:
                if not route.request.failure:
                    raise

        await page.route("**/api/answer/stream", reply)
        await page.route("**/api/display/analyze", emotion)
        results = []
        try:
            await page.goto(root)
            await page.wait_for_function("displaySync.registered")
            await display.set_viewport_size({"width": 800, "height": 480})
            await display.goto(root + "/display")
            await page.evaluate("() => {chatInput.value='受伤动物怎么样了？'; window.replyRun=sendChatMessage();}")
            await page.wait_for_function("qwenPlayer?.sources.size > 0", polling=20)
            assert await page.evaluate("isAnswering && displaySync.state === 'speaking' && displaySync.tone === 'happy'")
            await display.wait_for_function("eyes.dataset.tone === 'happy'", polling=30)
            assert await page.evaluate("isAnswering"), "Reply ended before the PCM buffers drained"
            await page.evaluate("window.replyRun")
            assert await page.evaluate("displaySync.state === 'success' && displaySync.tone === 'happy'")
            await page.evaluate("setWakeupStatus('继续对话：正在聆听问题', 'listening')")
            await page.wait_for_timeout(2000)
            assert await page.evaluate("displaySync.state === 'success' && displaySync.tone === 'happy'")
            await page.evaluate("""async () => {
              const previous = wakeupRecorder;
              const fake = {stopping: false}; wakeupRecorder = fake;
              try { await handleWakeupSocketMessage(fake, {data: JSON.stringify({type: 'speech_start'})}); }
              finally { wakeupRecorder = previous; }
            }""")
            assert await page.evaluate("displaySync.state === 'listening' && displaySync.tone === 'neutral'")
            await page.wait_for_timeout(1200)
            assert await page.evaluate("displaySync.state === 'listening'"), "Old hold timer restored a stale expression"
            results.append("Local tone and PCM start before delayed model; completion waits for drain; ready microphone preserves tone; actual speech clears immediately")

            await page.evaluate("() => {chatInput.value='再说一次'; window.replyRun=sendChatMessage();}")
            await page.wait_for_function("qwenPlayer?.sources.size > 0", polling=20)
            await page.evaluate("""() => {
              answerGeneration++; currentAnswerController?.abort();
              stopCurrentTTS(); isAnswering=false; chatSendBtn.disabled=false;
              setDisplayEmotion('listening', 0, true);
            }""")
            await page.wait_for_timeout(1000)
            assert await page.evaluate("displaySync.tone === 'neutral' && displaySync.state === 'listening' && !qwenPlayer")
            results.append("Interrupt stops PCM and stale emotion without resuming old playback")

            await page.evaluate("() => {autoSpeakEnabled=false; chatInput.value='你有疑问吗'; window.replyRun=sendChatMessage();}")
            await page.evaluate("window.replyRun")
            await display.wait_for_function("eyes.dataset.tone === 'curious'", polling=30)
            await page.wait_for_timeout(2000)
            assert await page.evaluate("displaySync.state === 'success' && displaySync.tone === 'curious'")
            await page.wait_for_function("displaySync.state === 'idle' && displaySync.tone === 'neutral'", polling=30)
            await page.evaluate("""() => {
              autoSpeakEnabled=true; window.sessionActions=[];
              scheduleWakeupResume=() => sessionActions.push({action:'resume', buffers:qwenPlayer?.sources.size || 0});
              scheduleWakeupContinuation=() => sessionActions.push({action:'continue', buffers:qwenPlayer?.sources.size || 0});
              chatInput.value='结束语流程测试'; window.replyRun=sendChatMessage(true);
            }""")
            await page.wait_for_function("qwenPlayer?.sources.size > 0", polling=20)
            assert await page.evaluate("sessionActions.length === 0"), "Ended session before farewell playback finished"
            await page.evaluate("window.replyRun")
            assert await page.evaluate("sessionActions.length === 1 && sessionActions[0].action === 'resume' && sessionActions[0].buffers === 0")
            await page.evaluate("() => {sessionActions=[]; chatInput.value='继续问保护河流'; window.replyRun=sendChatMessage(true);}")
            await page.evaluate("window.replyRun")
            assert await page.evaluate("sessionActions.length === 1 && sessionActions[0].action === 'continue' && sessionActions[0].buffers === 0")
            results.append("End signal returns to wake-word mode only after farewell PCM drains; normal turns continue listening")
            assert not tts_requests, tts_requests
            assert not errors, errors
            results.append("Text-only replies accept the added tones; separate reply TTS requests remain zero")
            cases = {'happy':'开心', 'concerned':'关切', 'sad':'难过', 'surprised':'惊讶',
                     'curious':'疑惑', 'shy':'害羞', 'proud':'自豪', 'encouraging':'鼓励'}
            for tone, label in cases.items():
                await page.evaluate("""label => {
                  autoSpeakEnabled=false; chatInput.value=`展示${label}的表情`; window.replyRun=sendChatMessage();
                }""", label)
                await page.evaluate('window.replyRun')
                await display.wait_for_function('tone => eyes.dataset.tone === tone', arg=tone, polling=30)
                assert await display.evaluate("""tone => new Set([0,4,8,12,20,30].map(n =>
                  toneRenderer.render(tone,n).toDataURL())).size > 1""", tone), f'{tone} has no animation'
            results.append('All eight requested expressions reach the display and each renders distinct animation frames')
            print(json.dumps({"passed": results, "reply_tts_requests": len(tts_requests), "page_errors": errors}, ensure_ascii=False, indent=2))
        except Exception:
            print(json.dumps({
                "ui": await page.evaluate("({state:displaySync.state,tone:displaySync.tone,registered:displaySync.registered,retired:displaySync.retired,session:displaySync.sessionId,inFlight:displaySync.inFlight,phase:displayWakeupPhase,transient:displayTransient,answering:isAnswering})"),
                "screen": await display.evaluate("({state:activeState,tone:eyes.dataset.tone,snapshot,polling,manifest:Boolean(manifest),frame:Boolean(currentFrame)})"),
                "errors": errors,
            }, ensure_ascii=False, indent=2))
            raise
        finally:
            await context.close()
            await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    asyncio.run(main(parser.parse_args().url.rstrip("/")))
