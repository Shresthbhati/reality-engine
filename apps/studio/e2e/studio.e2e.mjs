import { chromium } from 'playwright-core';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const studioUrl = process.env.STUDIO_URL;
const sampleImagePath = process.env.SAMPLE_IMAGE_PATH;
if (!studioUrl || !sampleImagePath) {
  throw new Error('STUDIO_URL and SAMPLE_IMAGE_PATH env vars are required');
}

const chromeCandidates = [
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  '/usr/bin/chromium',
  '/usr/bin/google-chrome',
];
const executablePath = chromeCandidates.find((p) => fs.existsSync(p));
if (!executablePath) {
  throw new Error('No Chromium/Chrome/Edge executable found for browser E2E');
}

console.error('[e2e] launching', executablePath);
const browser = await chromium.launch({
  executablePath,
  headless: true,
  args: ['--no-sandbox', '--disable-gpu-sandbox'],
});
console.error('[e2e] browser launched');

async function assertNoErrorBanner(page, stepName) {
  const err = await page.locator('[data-error="true"]').textContent().catch(() => null);
  assert.ok(!err, `${stepName} failed with UI error: ${err}`);
}

try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  page.setDefaultTimeout(15000);
  const pageErrors = [];
  page.on('pageerror', (err) => pageErrors.push(err.message));
  page.on('response', (res) => {
    if (res.url().includes('/api/')) console.error('[e2e] resp', res.status(), res.url());
  });

  console.error('[e2e] goto', studioUrl);
  await page.goto(studioUrl, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('#viewport canvas', { timeout: 8000 });
  const canvasBox = await page.locator('#viewport canvas').boundingBox();
  assert.ok(canvasBox && canvasBox.width > 100 && canvasBox.height > 100, '3D canvas did not render');
  console.error('[e2e] canvas ok');

  // 2. Sign up against real /api/auth/signup
  await page.fill('#auth-email', `studio_e2e_${Date.now()}@example.com`);
  await page.fill('#auth-pass', 'StrongPass123!');
  await page.click('#signup-btn');
  await page.waitForFunction(
    () =>
      ((document.querySelector('#status-bar')?.textContent ?? '').includes('Found') &&
        document.body.innerText.includes('Signed in:')) ||
      Boolean(document.querySelector('[data-error="true"]')),
    { timeout: 8000 },
  );
  await assertNoErrorBanner(page, 'signup');
  console.error('[e2e] signup ok');

  // 3. Create Project/World against real /api/worlds
  console.error('[e2e] filling world name');
  await page.fill('#new-world-name', 'E2E Studio Room');
  console.error('[e2e] clicking create-world-btn');
  await page.click('#create-world-btn');
  console.error('[e2e] waiting for world status');
  await page.waitForFunction(
    () => {
      const txt = document.querySelector('#status-bar')?.textContent ?? '';
      return txt.includes('entities') || txt.includes('0 versions yet') || Boolean(document.querySelector('[data-error="true"]'));
    },
    { timeout: 10000 },
  );
  await assertNoErrorBanner(page, 'createWorld');
  console.error('[e2e] world ok');

  // 4. Upload real image against real /api/sessions + /api/uploads + /api/worlds/{id}/attach
  await page.setInputFiles('#upload-files', path.resolve(sampleImagePath));
  await page.click('#upload-btn');
  await page.waitForFunction(
    () =>
      (document.querySelector('#status-bar')?.textContent ?? '').includes('Uploaded 1 image') ||
      Boolean(document.querySelector('[data-error="true"]')),
    { timeout: 8000 },
  );
  await assertNoErrorBanner(page, 'upload');
  console.error('[e2e] upload ok');

  // 5. Trigger reconstruction against real /api/reconstructions
  await page.click('#reconstruct-btn');
  await page.waitForFunction(
    () =>
      (document.querySelector('#status-bar')?.textContent ?? '').includes('Reconstruction') ||
      Boolean(document.querySelector('[data-error="true"]')),
    { timeout: 8000 },
  );
  await assertNoErrorBanner(page, 'reconstruct');
  console.error('[e2e] reconstruct ok');

  // 6. Check status endpoint
  await page.click('#status-btn');
  await page.waitForTimeout(400);

  assert.equal(pageErrors.length, 0, `Uncaught page errors: ${pageErrors.join('; ')}`);
  console.log('STUDIO_BROWSER_E2E_OK');
} finally {
  await browser.close();
}
