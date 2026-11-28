/**
 * Activity — the platform's real event log.
 *
 * Re-exports types and getActivity from the api module, plus the ApiError
 * helpers pages need to render a failure distinctly from an empty log.
 */
export {
  fetchActivity,
  getActivity,
  type ActivityEvent,
  type ActivityType,
} from "./api/activity";
export { isApiError, ApiError } from "./api/client";