import { test, expect } from "@playwright/test";
import { seed } from "./support";

const { worldId } = seed();

test.describe("World Adoption Review (Task U.7)", () => {
  test("reviews candidate capture sessions, displays 5-state changeset tabs and enforces Rule 18 manual edit protection", async ({
    page,
  }) => {
    // Navigate to adoptions page
    await page.goto(`/worlds/${worldId}/adoptions`);

    // Verify page header and adoption section
    await expect(page.getByRole("heading", { name: /Session Adoption Review/i })).toBeVisible();
    await expect(page.getByText(/Multi-Session World Adoption/i)).toBeVisible();

    // Route stubbing for adoption proposal
    await page.route(`**/api/worlds/${worldId}/adoptions`, async (route) => {
      if (route.request().method() === "POST") {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            adoption_id: "adp_test_suite_123",
            world_id: worldId,
            session_id: "ses_second_survey_02",
            parent_version_id: "v-001",
            alignment_confidence: 0.91,
            alignment_tier: "HIGH",
            change_set: {
              matched: ["wall_01", "floor_01"],
              new: ["room_kitchen_02", "wall_k01"],
              updated: ["wall_02"],
              uncertain: ["chair_obscured_03"],
              conflicting: ["wall_custom_edit_01"],
            },
            status: "PROPOSED",
            created_at: new Date().toISOString(),
            summary: {
              total_candidate_entities: 7,
              matched_count: 2,
              new_count: 2,
              updated_count: 1,
              uncertain_count: 1,
              conflicting_count: 1,
            },
          }),
        });
      } else {
        await route.continue();
      }
    });

    // Enter session id and trigger comparison
    const input = page.getByPlaceholder(/Enter survey Session ID/i);
    await input.fill("ses_second_survey_02");
    await page.getByRole("button", { name: /Compare & Review/i }).click();

    // Verify alignment score and tier badge
    await expect(page.getByText(/High Confidence/i)).toBeVisible();
    await expect(page.getByText(/Alignment: 91%/i)).toBeVisible();

    // Verify all 5-state category tabs are present with exact entity counts
    await expect(page.getByRole("button", { name: /New 2/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /Matched 2/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /Updated 1/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /Uncertain 1/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /Conflicting 1/i })).toBeVisible();

    // Switch to Conflicting tab to verify Rule 18 Protection UI
    await page.getByRole("button", { name: /Conflicting 1/i }).click();
    await expect(page.getByText("wall_custom_edit_01")).toBeVisible();
    await expect(page.getByText(/Rule 18 Protected/i)).toBeVisible();
    await expect(page.getByRole("button", { name: /Keep Manual Edit/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /Overwrite with Survey/i })).toBeVisible();

    // Route stubbing for commit
    await page.route(`**/api/worlds/${worldId}/adoptions/adp_test_suite_123/commit`, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          adoption_id: "adp_test_suite_123",
          world_id: worldId,
          version_id: "v-002-adopted",
          parent_version_id: "v-001",
          status: "ADOPTED",
          stats: { applied_entities: 5, preserved_edits: 1 },
          message: "Successfully adopted 5 entities into world model. Version incremented.",
        }),
      });
    });

    // Click commit button
    await page.getByRole("button", { name: /Adopt Changes into World/i }).click();

    // Verify success banner and new version
    await expect(page.getByText(/Session Successfully Adopted into World Model/i)).toBeVisible();
    await expect(page.getByText(/Committed Version: v-002-adopted/i)).toBeVisible();
  });
});
