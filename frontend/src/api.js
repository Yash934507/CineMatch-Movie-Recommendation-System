const API_BASE = import.meta.env?.VITE_API_URL || "/api/v1";
let unauthorizedHandler = null;

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function messageForStatus(status, detail, path, authRequired) {
  if (status === 401) {
    if (path === "/auth/login") return "Invalid email or password.";
    if (authRequired) return "Your session has ended. Sign in again to continue.";
  }
  if (status === 403) return "This action is not allowed from this page.";
  if (status === 409) return typeof detail === "string" ? detail : "An account with this email already exists.";
  if (status === 422) {
    const messages = Array.isArray(detail)
      ? detail.map((item) => item.msg).filter(Boolean)
      : [];
    return messages.length ? messages.join(". ") : "Check the information you entered and try again.";
  }
  if (status === 429) return "Too many attempts. Please wait a moment and try again.";
  if (status >= 500) return "CineMatch is having trouble right now. Please try again shortly.";
  return typeof detail === "string" ? detail : `Request failed (${status}).`;
}

async function request(path, { signal, body, authRequired = false, ...options } = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...options,
      signal,
      body: body === undefined ? undefined : JSON.stringify(body),
      credentials: authRequired || path.startsWith("/auth/") ? "include" : "omit",
      headers: {
        ...(body === undefined ? {} : { "Content-Type": "application/json" }),
        ...options.headers,
      },
    });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new ApiError("Unable to reach CineMatch. Check your connection and try again.", 0);
  }

  if (response.status === 401 && authRequired && unauthorizedHandler) {
    unauthorizedHandler();
  }
  if (!response.ok) {
    let payload = {};
    try {
      payload = await response.json();
    } catch {
      payload = {};
    }
    throw new ApiError(
      messageForStatus(response.status, payload.detail, path, authRequired),
      response.status,
    );
  }
  if (response.status === 204) return null;
  try {
    return await response.json();
  } catch {
    throw new ApiError("CineMatch returned an unexpected response. Please refresh and try again.", response.status);
  }
}

export function setUnauthorizedHandler(handler) {
  unauthorizedHandler = handler;
}

export const api = {
  health: (signal) => request("/health", { signal }),
  register: (fields) => request("/auth/register", { method: "POST", body: fields }),
  login: (fields) => request("/auth/login", { method: "POST", body: fields }),
  logout: () => request("/auth/logout", { method: "POST" }),
  me: () => request("/auth/me"),
  recommendations: (userId, limit = 10, signal) =>
    request(`/demo/recommendations/${encodeURIComponent(userId)}?limit=${limit}`, { signal }),
  demoPrediction: (userId, movieId, signal) =>
    request(`/demo/predictions/${encodeURIComponent(userId)}/${encodeURIComponent(movieId)}`, { signal }),
  accountRecommendations: (limit = 10, signal) =>
    request(`/me/recommendations?limit=${limit}`, { signal, authRequired: true }),
  accountPrediction: (movieId, signal) =>
    request(`/me/predictions/${encodeURIComponent(movieId)}`, { signal, authRequired: true }),
  rateMovie: (userId, movieId, rating) =>
    request(`/users/${encodeURIComponent(userId)}/ratings`, {
      method: "POST",
      body: { movie_id: movieId, rating },
      authRequired: true,
    }),
  addToWatchlist: (userId, movieId) =>
    request(`/users/${encodeURIComponent(userId)}/watchlist`, {
      method: "POST",
      body: { movie_id: movieId },
      authRequired: true,
    }),
  submitFeedback: (userId, movieId, feedbackType) =>
    request(`/users/${encodeURIComponent(userId)}/feedback`, {
      method: "POST",
      body: { movie_id: movieId, feedback_type: feedbackType },
      authRequired: true,
    }),
  onboardingRecommendations: (ratings, limit = 10, signal) =>
    request("/onboarding/recommendations", {
      signal,
      method: "POST",
      body: { ratings, limit },
    }),
  searchMovies: ({ query = "", genre = "", limit = 20 }, signal) => {
    const params = new URLSearchParams({ q: query, limit: String(limit) });
    if (genre) params.set("genre", genre);
    return request(`/movies?${params}`, { signal });
  },
  movie: (movieId, signal) => request(`/movies/${movieId}`, { signal }),
  similarMovies: (movieId, limit = 6, signal) =>
    request(`/movies/${movieId}/similar?limit=${limit}`, { signal }),
};

export { API_BASE };
