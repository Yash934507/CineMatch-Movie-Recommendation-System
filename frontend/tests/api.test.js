import assert from "node:assert/strict";
import { afterEach, test } from "node:test";

import { ApiError, api, setUnauthorizedHandler } from "../src/api.js";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
  setUnauthorizedHandler(null);
});

function respond(payload, status = 200) {
  return new Response(payload === null ? null : JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

test("authentication requests use backend field names and credentialed cookies", async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return respond({ id: 8, email: "viewer@example.test", display_name: "Viewer" }, url.endsWith("/auth/register") ? 201 : 200);
  };

  await api.register({ display_name: "Viewer", email: "viewer@example.test", password: "a secure password" });
  await api.login({ email: "viewer@example.test", password: "a secure password" });
  await api.me();

  assert.deepEqual(calls.map(({ url }) => url), [
    "/api/v1/auth/register",
    "/api/v1/auth/login",
    "/api/v1/auth/me",
  ]);
  assert.equal(JSON.parse(calls[0].options.body).display_name, "Viewer");
  assert.equal(JSON.parse(calls[0].options.body).email, "viewer@example.test");
  assert.equal(JSON.parse(calls[1].options.body).password, "a secure password");
  for (const { options } of calls) {
    assert.equal(options.credentials, "include");
    assert.equal(options.headers.Cookie, undefined);
  }
});

test("logout accepts an empty 204 response", async () => {
  let call;
  globalThis.fetch = async (url, options) => {
    call = { url, options };
    return respond(null, 204);
  };

  assert.equal(await api.logout(), null);
  assert.equal(call.url, "/api/v1/auth/logout");
  assert.equal(call.options.method, "POST");
});

test("personal account endpoints never use the public MovieLens identity routes", async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return respond(url.includes("/predictions/") ? { movie_id: 260 } : { recommendations: [] });
  };

  await api.accountRecommendations();
  await api.accountPrediction(260);
  await api.rateMovie(8, 260, 4.5);
  await api.addToWatchlist(8, 260);
  await api.submitFeedback(8, 260, "like");

  assert.deepEqual(calls.map(({ url }) => url), [
    "/api/v1/me/recommendations?limit=10",
    "/api/v1/me/predictions/260",
    "/api/v1/users/8/ratings",
    "/api/v1/users/8/watchlist",
    "/api/v1/users/8/feedback",
  ]);
  assert.ok(calls.every(({ options }) => options.credentials === "include"));
});

test("demo recommendation and prediction requests use explicit public routes", async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return respond({});
  };

  await api.recommendations(12, 5);
  await api.demoPrediction(12, 260);

  assert.deepEqual(calls.map(({ url }) => url), [
    "/api/v1/demo/recommendations/12?limit=5",
    "/api/v1/demo/predictions/12/260",
  ]);
  assert.ok(calls.every(({ options }) => options.credentials === "omit"));
});

test("API errors map backend status codes to friendly messages", async (t) => {
  const cases = [
    [401, "backend detail"],
    [403, "This action is not allowed"],
    [409, "An account with this email already exists"],
    [422, "Check the information you entered"],
    [429, "Too many attempts"],
    [500, "CineMatch is having trouble"],
  ];

  for (const [status, message] of cases) {
    await t.test(String(status), async () => {
      globalThis.fetch = async () => respond({ detail: status === 409 ? message : "backend detail" }, status);
      await assert.rejects(api.health(), (error) => {
        assert.ok(error instanceof ApiError);
        assert.equal(error.status, status);
        assert.match(error.message, new RegExp(message));
        return true;
      });
    });
  }
});

test("422 field errors are presented without exposing request values", async () => {
  globalThis.fetch = async () => respond({
    detail: [{ loc: ["body", "password"], msg: "Password is invalid", type: "value_error" }],
  }, 422);

  await assert.rejects(api.register({
    display_name: "Viewer",
    email: "viewer@example.test",
    password: "private input",
  }), (error) => {
    assert.equal(error.status, 422);
    assert.equal(error.message, "Password is invalid");
    assert.equal(error.message.includes("private input"), false);
    return true;
  });
});

test("protected 401 responses notify the app to clear authenticated state", async () => {
  let cleared = false;
  setUnauthorizedHandler(() => { cleared = true; });
  globalThis.fetch = async () => respond({ detail: "Authentication required" }, 401);

  await assert.rejects(api.accountRecommendations(), (error) => {
    assert.equal(error.status, 401);
    assert.equal(error.message, "Your session has ended. Sign in again to continue.");
    return true;
  });
  assert.equal(cleared, true);
});

test("invalid login keeps the generic credential error instead of implying session expiry", async () => {
  globalThis.fetch = async () => respond({ detail: "Invalid email or password" }, 401);

  await assert.rejects(api.login({
    email: "viewer@example.test",
    password: "incorrect password",
  }), (error) => {
    assert.equal(error.status, 401);
    assert.equal(error.message, "Invalid email or password.");
    assert.equal(error.message.includes("session"), false);
    return true;
  });
});

test("network failures have a safe user-facing message", async () => {
  globalThis.fetch = async () => { throw new TypeError("fetch failed"); };

  await assert.rejects(api.health(), (error) => {
    assert.equal(error.status, 0);
    assert.equal(error.message, "Unable to reach CineMatch. Check your connection and try again.");
    return true;
  });
});
