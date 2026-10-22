"""Verify real canvas animation, clipping and preview controls without cloud calls."""
from __future__ import annotations

import argparse
import asyncio
import io
import json
from pathlib import Path

from PIL import Image
from playwright.async_api import async_playwright


async def main(root: str) -> None:
    output = Path("data/runtime")
    output.mkdir(parents=True, exist_ok=True)
    asset_dir = Path("src/rag_system/web/assets/emotions")
    bounds = json.loads((asset_dir / "manifest.json").read_text(encoding="utf-8"))["layout"]["content_bounds"]
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
            args=["--disable-background-timer-throttling", "--disable-renderer-backgrounding"])
        context = await browser.new_context(viewport={"width": 1000, "height": 900})
        page = await context.new_page()
        display = await context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        display.on("pageerror", lambda error: errors.append(str(error)))
        results = []

        async def post(path, data):
            response = await context.request.post(root + path, data=data)
            assert response.ok, (path, await response.text())
            return await response.json()

        try:
            before = await (await context.request.get(root + "/api/display/state")).json()
            await page.goto(root + "/display/preview")
            await page.wait_for_function("document.querySelectorAll('canvas').length === 8")
            after = await (await context.request.get(root + "/api/display/state")).json()
            assert before == after, "Opening the gallery changed the live display"

            rendered = await page.evaluate("""bounds => {
              const rows = [];
              for (const tone of Object.keys(DisplayToneRenderer.tones)) {
                const hashes = new Set();
                let visibleFrames = 0;
                for (let frame = 0; frame < 40; frame++) {
                  const canvas = renderer.render(tone, frame);
                  hashes.add(canvas.toDataURL());
                  const pixels = canvas.getContext('2d').getImageData(0, 0, 96, 48).data;
                  let visible = 0;
                  for (let y = 0; y < 48; y++) for (let x = 0; x < 96; x++) {
                    const i = (y * 96 + x) * 4;
                    if (pixels[i] + pixels[i+1] + pixels[i+2] === 0) continue;
                    visible++;
                    const [left, top, right, bottom] = bounds[x < 48 ? 0 : 1];
                    if (x < left || x > right || y < top || y > bottom)
                      throw new Error(tone + ' frame ' + frame + ' clips at ' + x + ',' + y);
                  }
                  if (visible) visibleFrames++;
                }
                if (hashes.size < 4 || visibleFrames !== 40) throw new Error(tone + ' is static or blank');
                rows.push({tone, distinctFrames: hashes.size, visibleFrames});
              }
              return rows;
            }""", bounds)
            assert len(rendered) == 8
            results.append({"animation_frames_and_safe_bounds": rendered})

            # Export deterministic frames from the actual gallery, rather than
            # photographing an arbitrarily timed blink.
            await page.evaluate("clearInterval(galleryTimer)")
            frames = []
            for frame in range(0, 40, 2):
                await page.evaluate("frame => drawGallery(frame)", frame)
                shot = await page.screenshot(full_page=True)
                frames.append(Image.open(io.BytesIO(shot)).convert("RGB"))
            frames[0].save(output / "emotion-gallery.png")
            frames[0].save(output / "emotion-gallery.gif", save_all=True, append_images=frames[1:],
                duration=200, loop=0, disposal=2)

            await display.set_viewport_size({"width": 800, "height": 480})
            await display.goto(root + "/display")
            await display.wait_for_function("Boolean(currentFrame)")
            await post("/api/display/mode", {"mode": "emotion"})
            for tone in (row["tone"] for row in rendered):
                await post("/api/display/emotion", {"state": "speaking", "tone": tone})
                await display.wait_for_function("tone => eyes.dataset.tone === tone", arg=tone)
                images = set()
                for _ in range(5):
                    images.add(await display.evaluate("eyes.toDataURL()"))
                    await display.wait_for_timeout(220)
                assert len(images) > 1, tone + " stopped animating while the server state stayed unchanged"
            results.append("All 8 expressions animate at 800×480 without new server state events")
            await post("/api/display/emotion", {"state": "thinking", "tone": "happy"})
            await display.wait_for_function("eyes.dataset.state === 'thinking' && eyes.dataset.tone === 'neutral' && toneTimer === null")
            await post("/api/display/emotion", {"state": "speaking", "tone": "happy"})
            await display.wait_for_function("eyes.dataset.tone === 'happy'")
            await post("/api/display/mode", {"mode": "web"})
            await display.wait_for_function("toneTimer === null && webPanel.style.display === 'block'")
            results.append("Thinking and web mode stop the content-tone animation")

            await page.get_by_role("button", name="预览疑惑到外接屏", exact=True).click()
            await display.wait_for_function("eyes.dataset.tone === 'curious' && eyes.dataset.state === 'speaking'")
            await page.get_by_role("button", name="恢复待机", exact=True).click()
            await display.wait_for_function("eyes.dataset.state === 'idle' && eyes.dataset.tone === 'neutral' && toneTimer === null")
            results.append("Gallery controls can preview on the display and restore idle")
            assert not errors, errors
            print(json.dumps({"passed": results, "page_errors": errors}, ensure_ascii=False, indent=2))
        finally:
            await post("/api/display/mode", {"mode": "emotion"})
            await post("/api/display/emotion", {"state": "idle", "tone": "neutral"})
            await context.close()
            await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    asyncio.run(main(parser.parse_args().url.rstrip("/")))
