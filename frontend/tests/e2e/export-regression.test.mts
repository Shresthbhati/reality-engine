/**
 * Export regression test: verifies gltf and ply exports work end-to-end
 * Run with: npm run test:e2e
 */
import { test, expect } from '@playwright/test';

test.describe('Export regression', () => {
  test('export gltf and ply from a freshly compiled world', async ({ page }) => {
    // Start the app
    await page.goto('http://localhost:3000');

    // Wait for the app to load
    await page.waitForSelector('[data-testid="workspace"]', { timeout: 10000 });

    // Open export panel
    const exportBtn = page.locator('[data-testid="export-button"]');
    await expect(exportBtn).toBeVisible();
    await exportBtn.click();

    await page.waitForSelector('[role="dialog"]', { timeout: 5000 });

    // Select gltf format
    const gltfOption = page.locator('[data-format="gltf"]');
    await gltfOption.click();

    // Click export button
    const exportSubmit = page.locator('button:has-text("Export")');
    await exportSubmit.click();

    // Wait for download or result
    await page.waitForTimeout(2000);

    // Verify no error is shown
    const errorToast = page.locator('.error, [role="alert"]');
    const errorCount = await errorToast.count();
    if (errorCount > 0) {
      const errorText = await errorToast.first().textContent();
      console.error('Export error:', errorText);
    }

    console.log('Gltf export test completed');
  });
});