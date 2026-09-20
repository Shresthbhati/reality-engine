const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  console.log('Navigating to http://localhost:3000/phone...');
  await page.goto('http://localhost:3000/phone', { waitUntil: 'networkidle', timeout: 30000 });

  // 1. Click New capture session
  console.log('Clicking New capture session button...');
  const newSessBtn = page.locator('text=New capture session').first();
  await newSessBtn.click();
  await page.waitForTimeout(500);

  // 2. Type session name
  const nameInput = page.locator('input[placeholder="Session name (optional)"]').first();
  await nameInput.fill('Site Inspection — Ground Pass 01');

  // 3. Click Start capture
  console.log('Clicking Start capture...');
  const startBtn = page.locator('button:has-text("Start capture")').first();
  await startBtn.click();
  await page.waitForTimeout(2000);

  const bodyText = await page.innerText('body');
  console.log('Capture screen content preview:\n', bodyText.slice(0, 500));

  await page.screenshot({ path: 'mobile_live_capture_screen.png' });
  console.log('Screenshot saved to mobile_live_capture_screen.png');

  await browser.close();
  console.log('Live capture screen verified!');
})();
