/**
 * Minimal stand-in for `next/server`, sufficient for the proxy routes under
 * contract test. Only the surface those routes touch is implemented.
 */

/** Mirrors NextResponse.json: a Response carrying a JSON body and a status. */
export class NextResponse extends Response {
  static json(body, init) {
    return new Response(body === undefined ? null : JSON.stringify(body), {
      status: init?.status ?? 200,
      headers: { "content-type": "application/json" },
    });
  }
}

/** The routes only read `request.url`; the rest of the surface is unused. */
export class NextRequest extends Request {}
