const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  console.log('Navigating to http://localhost:3000/viewer...');
  await page.goto('http://localhost:3000/viewer', { waitUntil: 'networkidle', timeout: 30000 });

  // 1. Click Kolkata world row to load backend world
  const worldRow = page.locator('div').filter({ hasText: /^Kolkata Survey Site/ }).first();
  const worldDetailPromise = page.waitForResponse(r => r.url().includes('/api/world/v-kolkata-01') && !r.url().includes('/points'), { timeout: 15000 });
  await worldRow.click();
  await worldDetailPromise;
  await page.waitForTimeout(2000);

  // 2. Switch to ENTITIES tab
  const entitiesTab = page.locator('text=ENTITIES').first();
  await entitiesTab.click();
  await page.waitForTimeout(1000);

  // 3. Click on "struct plane 007"
  console.log('Finding "struct plane 007" row...');
  const planeRow = page.locator('text=struct plane 007').first();
  await planeRow.click();
  await page.waitForTimeout(1500);

  // 4. Save screenshot with struct plane 007 selected
  await page.screenshot({ path: 'viewer_e2e_struct_plane_selected.png' });
  console.log('Screenshot saved to viewer_e2e_struct_plane_selected.png');

  // Verify Inspector contents
  const bodyText = await page.innerText('body');
  console.log('Contains struct-plane-007:', bodyText.includes('struct-plane-007') || bodyText.includes('struct plane 007'));
  console.log('Contains floor:', bodyText.includes('floor') || bodyText.includes('FLOOR'));
  console.log('Contains geom-struct-plane-007:', bodyText.includes('geom-struct-plane-007'));

  await browser.close();
  console.log('E2E selection verification completed!');
})();
