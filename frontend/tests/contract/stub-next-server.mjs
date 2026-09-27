/**
 * Module-resolution hook that redirects `next/server` to a minimal stub.
 *
 * The proxy route under test only needs `NextResponse.json`; the Next runtime
 * itself is irrelevant to the contract being asserted, and importing the real
 * `next/server` outside a Next build fails to resolve under plain Node.
 *
 * Used by the sessions proxy contract harness (see sessions-proxy.contract.mjs)
 * and by tests/test_frontend_sessions_proxy_contract.py.
 */
const STUB = new URL("./next-server-stub.mjs", import.meta.url).href;

export function resolve(specifier, context, nextResolve) {
  if (specifier === "next/server") {
    return { url: STUB, shortCircuit: true };
  }
  return nextResolve(specifier, context);
}
