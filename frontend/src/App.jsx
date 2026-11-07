import { useCallback, useEffect, useMemo, useState } from "react";
import { api, setUnauthorizedHandler } from "./api";

const READ_ONLY_DEMO = import.meta.env?.VITE_CINEMATCH_READ_ONLY_DEMO === "true";
const GENRES = ["All genres", "Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi", "Thriller"];
const PALETTES = [
  ["#7c3aed", "#db2777"],
  ["#0369a1", "#06b6d4"],
  ["#c2410c", "#f59e0b"],
  ["#0f766e", "#34d399"],
  ["#4338ca", "#8b5cf6"],
  ["#be123c", "#fb7185"],
];

function Icon({ name, size = 20 }) {
  const paths = {
    search: <><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></>,
    sparkles: <><path d="m12 3 1.4 4.1L17.5 8.5l-4.1 1.4L12 14l-1.4-4.1-4.1-1.4 4.1-1.4L12 3Z"/><path d="m19 14 .8 2.2L22 17l-2.2.8L19 20l-.8-2.2L16 17l2.2-.8L19 14Z"/></>,
    star: <path d="m12 2 3 6 6.5 1-4.7 4.6 1.1 6.4-5.9-3.1L6.1 20l1.1-6.4L2.5 9 9 8l3-6Z"/>,
    play: <path d="m8 5 11 7-11 7V5Z"/>,
    arrow: <><path d="M5 12h14"/><path d="m13 6 6 6-6 6"/></>,
    close: <><path d="m6 6 12 12"/><path d="M18 6 6 18"/></>,
    film: <><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M7 5v14M17 5v14M3 9h4M17 9h4M3 15h4M17 15h4"/></>,
    user: <><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></>,
  };
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{paths[name]}</svg>;
}

function scoreLabel(value) {
  return Number(value || 0).toFixed(1);
}

function MovieArtwork({ movie, compact = false }) {
  const palette = PALETTES[Math.abs(movie.movie_id) % PALETTES.length];
  const words = movie.title.replace(/\([^)]*\)/g, "").trim().split(/\s+/);
  const initials = words.slice(0, 2).map((word) => word[0]).join("").toUpperCase();
  return (
    <div className={`artwork ${compact ? "artwork--compact" : ""}`} style={{ "--from": palette[0], "--to": palette[1] }}>
      <div className="artwork__grain" />
      <span className="artwork__initials">{initials}</span>
      <Icon name="film" size={compact ? 22 : 28} />
    </div>
  );
}

function MovieCard({ movie, onSelect, rank }) {
  return (
    <button className="movie-card" onClick={() => onSelect(movie)} aria-label={`View ${movie.title}`}>
      <div className="movie-card__visual">
        <MovieArtwork movie={movie} />
        {rank && <span className="rank">{String(rank).padStart(2, "0")}</span>}
        {movie.predicted_rating != null && (
          <span className="score"><Icon name="star" size={13} /> {scoreLabel(movie.predicted_rating)}</span>
        )}
        <span className="movie-card__action"><Icon name="play" size={18} /></span>
      </div>
      <span className="movie-card__title">{movie.title}</span>
      <span className="movie-card__genres">{movie.genres.slice(0, 2).join(" · ") || "Uncategorized"}</span>
    </button>
  );
}

function SkeletonGrid({ count = 5 }) {
  return <div className="movie-grid">{Array.from({ length: count }, (_, index) => <div className="skeleton-card" key={index}><div /><span /><span /></div>)}</div>;
}

function EmptyState({ title, message }) {
  return <div className="empty-state"><span><Icon name="film" size={30} /></span><h3>{title}</h3><p>{message}</p></div>;
}

function TasteOnboarding({ movies, ratings, onRate, onSubmit, loading, onClose }) {
  const ratingCount = Object.keys(ratings).length;
  return (
    <section className="onboarding" id="onboarding">
      <div className="onboarding__heading">
        <div><span className="eyebrow">New viewer setup</span><h2>Rate movies you know</h2><p>Choose at least three. CineMatch estimates a temporary taste vector from the trained SVD factors.</p></div>
        <button className="onboarding__close" onClick={onClose} aria-label="Close taste setup"><Icon name="close" /></button>
      </div>
      <div className="rating-grid">
        {movies.map((movie) => (
          <article className="rating-card" key={movie.movie_id}>
            <MovieArtwork movie={movie} compact />
            <h3 title={movie.title}>{movie.title}</h3>
            <div className="star-picker" aria-label={`Rate ${movie.title}`}>
              {[1, 2, 3, 4, 5].map((value) => (
                <button key={value} className={(ratings[movie.movie_id] || 0) >= value ? "active" : ""} onClick={() => onRate(movie.movie_id, value)} aria-label={`${value} star${value === 1 ? "" : "s"}`}><Icon name="star" size={17} /></button>
              ))}
            </div>
          </article>
        ))}
      </div>
      <div className="onboarding__action"><span>{ratingCount} rated · minimum 3</span><button disabled={ratingCount < 3 || loading} onClick={onSubmit}>{loading ? "Building your profile…" : "Create my recommendations"}<Icon name="sparkles" size={17} /></button></div>
    </section>
  );
}

function AuthDialog({ onClose, onAuthenticated }) {
  const [mode, setMode] = useState("login");
  const [displayName, setDisplayName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (event) => {
    event.preventDefault();
    setError("");
    setNotice("");
    if (mode === "register" && password !== confirmPassword) {
      setError("Those passwords do not match.");
      return;
    }

    setBusy(true);
    try {
      if (mode === "register") {
        await api.register({ display_name: displayName.trim(), email: email.trim(), password });
        setPassword("");
        setConfirmPassword("");
        setMode("login");
        setNotice("Your account is ready. Sign in to start building your personal taste profile.");
      } else {
        const account = await api.login({ email: email.trim(), password });
        setPassword("");
        onAuthenticated(account);
      }
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="modal-backdrop auth-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="auth-dialog" role="dialog" aria-modal="true" aria-labelledby="auth-title">
        <button className="modal-close" onClick={onClose} aria-label="Close sign in"><Icon name="close" /></button>
        <span className="eyebrow"><Icon name="sparkles" size={15} /> Your own cinema</span>
        <h2 id="auth-title">{mode === "register" ? "Create your account" : "Welcome back"}</h2>
        <p className="auth-intro">{mode === "register" ? "Save your ratings and get recommendations shaped around your taste." : "Sign in to pick up where your next favorite left off."}</p>
        {notice && <p className="auth-notice" role="status">{notice}</p>}
        {error && <p className="auth-error" role="alert">{error}</p>}
        <form className="auth-form" onSubmit={submit}>
          {mode === "register" && (
            <label>Display name
              <input autoComplete="name" value={displayName} onChange={(event) => setDisplayName(event.target.value)} maxLength={80} required />
            </label>
          )}
          <label>Email
            <input type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} maxLength={320} required />
          </label>
          <label>Password
            <input type="password" autoComplete={mode === "register" ? "new-password" : "current-password"} value={password} onChange={(event) => setPassword(event.target.value)} minLength={mode === "register" ? 12 : 1} maxLength={1024} required />
          </label>
          {mode === "register" && (
            <label>Confirm password
              <input type="password" autoComplete="new-password" value={confirmPassword} onChange={(event) => setConfirmPassword(event.target.value)} minLength={12} maxLength={1024} required />
            </label>
          )}
          {mode === "register" && <p className="auth-hint">Use at least 12 characters. Your password is never stored in this browser.</p>}
          <button className="auth-submit" type="submit" disabled={busy}>{busy ? "One moment…" : mode === "register" ? "Create account" : "Sign in"}<Icon name="arrow" size={17} /></button>
        </form>
        <p className="auth-switch">
          {mode === "register" ? "Already have an account?" : "New to CineMatch?"}{" "}
          <button onClick={() => { setError(""); setNotice(""); setMode(mode === "register" ? "login" : "register"); }}> {mode === "register" ? "Sign in" : "Create an account"}</button>
        </p>
      </section>
    </div>
  );
}

function MovieModal({ movie, account, demoUserId, readOnlyDemo, onClose, onLoginRequired, onRatingSaved }) {
  const [similar, setSimilar] = useState([]);
  const [loading, setLoading] = useState(true);
  const [prediction, setPrediction] = useState(null);
  const [rating, setRating] = useState("4");
  const [actionBusy, setActionBusy] = useState(false);
  const [actionMessage, setActionMessage] = useState("");
  const [actionError, setActionError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    api.similarMovies(movie.movie_id, 6, controller.signal)
      .then(setSimilar)
      .catch(() => setSimilar([]))
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [movie.movie_id]);

  useEffect(() => {
    const controller = new AbortController();
    setPrediction(null);
    const predictionRequest = account
      ? api.accountPrediction(movie.movie_id, controller.signal)
      : api.demoPrediction(demoUserId, movie.movie_id, controller.signal);
    predictionRequest
      .then(setPrediction)
      .catch((error) => {
        if (error.name !== "AbortError" && error.status !== 409 && error.status !== 401) {
          setActionError(error.message);
        }
      });
    return () => controller.abort();
  }, [movie.movie_id, account, demoUserId]);

  useEffect(() => {
    const close = (event) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [onClose]);

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="movie-modal" role="dialog" aria-modal="true" aria-labelledby="modal-title">
        <button className="modal-close" onClick={onClose} aria-label="Close movie details"><Icon name="close" /></button>
        <div className="modal-hero">
          <MovieArtwork movie={movie} />
          <div>
            <span className="eyebrow">CineMatch selection</span>
            <h2 id="modal-title">{movie.title}</h2>
            <div className="modal-meta">
              {movie.mean_rating != null && <span><Icon name="star" size={15} /> {scoreLabel(movie.mean_rating)} community</span>}
              <span>{movie.rating_count.toLocaleString()} ratings</span>
            </div>
            <div className="genre-list">{movie.genres.map((genre) => <span key={genre}>{genre}</span>)}</div>
            {(prediction?.predicted_rating ?? movie.predicted_rating) != null && <p className="match-copy">{account ? "Your account prediction" : `MovieLens demo prediction for sample user ${demoUserId}`} is <strong>{scoreLabel(prediction?.predicted_rating ?? movie.predicted_rating)} / 5</strong>{account ? ", based on your account ratings." : ", based only on public sample data."}</p>}
          </div>
        </div>
        <div className="account-actions">
          <div className="section-heading section-heading--small"><div><span className="eyebrow">Your account</span><h3>Make it personal</h3></div></div>
          {account ? (
            <>
              <div className="account-actions__controls">
                <label>Rate this movie
                  <select value={rating} onChange={(event) => setRating(event.target.value)}>
                    {[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5].map((value) => <option key={value} value={value}>{value} / 5</option>)}
                  </select>
                </label>
                <button disabled={actionBusy} onClick={async () => {
                  setActionBusy(true); setActionMessage(""); setActionError("");
                  try { await api.rateMovie(account.id, movie.movie_id, Number(rating)); setActionMessage("Rating saved to your account."); onRatingSaved(); }
                  catch (error) { setActionError(error.message); }
                  finally { setActionBusy(false); }
                }}>Save rating</button>
                <button disabled={actionBusy} onClick={async () => {
                  setActionBusy(true); setActionMessage(""); setActionError("");
                  try { await api.addToWatchlist(account.id, movie.movie_id); setActionMessage("Added to your watchlist."); }
                  catch (error) { setActionError(error.message); }
                  finally { setActionBusy(false); }
                }}>Add to watchlist</button>
                <button disabled={actionBusy} onClick={async () => {
                  setActionBusy(true); setActionMessage(""); setActionError("");
                  try { await api.submitFeedback(account.id, movie.movie_id, "like"); setActionMessage("Added to your liked movies."); }
                  catch (error) { setActionError(error.message); }
                  finally { setActionBusy(false); }
                }}>Like</button>
                <button disabled={actionBusy} onClick={async () => {
                  setActionBusy(true); setActionMessage(""); setActionError("");
                  try { await api.submitFeedback(account.id, movie.movie_id, "dislike"); setActionMessage("Feedback saved."); }
                  catch (error) { setActionError(error.message); }
                  finally { setActionBusy(false); }
                }}>Not for me</button>
              </div>
              {actionMessage && <p className="auth-notice" role="status">{actionMessage}</p>}
              {actionError && <p className="auth-error" role="alert">{actionError}</p>}
            </>
          ) : readOnlyDemo ? (
            <p className="account-prompt">Account actions are disabled in this public read-only demo. Browse the catalog and MovieLens sample recommendations without creating an account.</p>
          ) : (
            <p className="account-prompt">Sign in to save ratings, build a watchlist, and get recommendations based on your account. <button onClick={onLoginRequired}>Sign in</button></p>
          )}
        </div>
        <div className="similar-block">
          <div className="section-heading section-heading--small"><div><span className="eyebrow">In the same orbit</span><h3>Similar movies</h3></div></div>
          {loading ? <div className="similar-loading">Finding close matches…</div> : similar.length ? (
            <div className="similar-row">{similar.map((item) => <div className="similar-item" key={item.movie_id}><MovieArtwork movie={item} compact /><span>{item.title}</span><small>{Math.round(item.similarity * 100)}% similar</small></div>)}</div>
          ) : <p className="muted">No close matches available for this title.</p>}
        </div>
      </section>
    </div>
  );
}

export default function App() {
  const [account, setAccount] = useState(null);
  const [authReady, setAuthReady] = useState(false);
  const [authOpen, setAuthOpen] = useState(false);
  const [demoUserId, setDemoUserId] = useState("1");
  const [recommendations, setRecommendations] = useState([]);
  const [strategy, setStrategy] = useState("");
  const [coldStart, setColdStart] = useState(false);
  const [recommendationSource, setRecommendationSource] = useState("demo");
  const [loadingRecommendations, setLoadingRecommendations] = useState(true);
  const [query, setQuery] = useState("");
  const [genre, setGenre] = useState("All genres");
  const [movies, setMovies] = useState([]);
  const [loadingMovies, setLoadingMovies] = useState(true);
  const [selectedMovie, setSelectedMovie] = useState(null);
  const [error, setError] = useState("");
  const [apiOnline, setApiOnline] = useState(null);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [onboardingMovies, setOnboardingMovies] = useState([]);
  const [onboardingRatings, setOnboardingRatings] = useState({});
  const [onboardingLoading, setOnboardingLoading] = useState(false);

  const loadRecommendations = useCallback((source, signal) => {
    setLoadingRecommendations(true);
    setRecommendations([]);
    setError("");
    const request = source.type === "account"
      ? api.accountRecommendations(10, signal)
      : api.recommendations(source.userId, 10, signal);
    request
      .then((payload) => {
        setRecommendations(payload.recommendations);
        setStrategy(payload.strategy);
        setColdStart(payload.cold_start);
        setRecommendationSource(source.type);
        setApiOnline(true);
      })
      .catch((requestError) => {
        if (requestError.name !== "AbortError") {
          setError(requestError.message);
          if (requestError.status === 0) setApiOnline(false);
        }
      })
      .finally(() => {
        if (!signal?.aborted) setLoadingRecommendations(false);
      });
  }, []);

  const loadMovies = useCallback(() => {
    setLoadingMovies(true);
    api.searchMovies({ query, genre: genre === "All genres" ? "" : genre, limit: 20 })
      .then((payload) => { setMovies(payload); setApiOnline(true); })
      .catch((requestError) => { setError(requestError.message); setApiOnline(false); })
      .finally(() => setLoadingMovies(false));
  }, [query, genre]);

  useEffect(() => {
    let active = true;
    setUnauthorizedHandler(() => {
      if (!active) return;
      setAccount(null);
      setRecommendations([]);
      setSelectedMovie(null);
      setOnboardingRatings({});
      setError("Your session has ended. Sign in again to continue.");
    });
    api.health().then(() => { if (active) setApiOnline(true); }).catch(() => { if (active) setApiOnline(false); });
    api.searchMovies({ limit: 20 })
      .then((payload) => {
        if (active) {
          setMovies(payload);
          setOnboardingMovies(payload.slice(0, 10));
        }
      })
      .catch((requestError) => { if (active) setError(requestError.message); })
      .finally(() => { if (active) setLoadingMovies(false); });
    api.me()
      .then((currentAccount) => { if (active) setAccount(currentAccount); })
      .catch((requestError) => {
        if (active && requestError.status !== 401) {
          setError(requestError.message);
          if (requestError.status === 0) setApiOnline(false);
        }
      })
      .finally(() => { if (active) setAuthReady(true); });
    return () => {
      active = false;
      setUnauthorizedHandler(null);
    };
  }, []);

  useEffect(() => {
    if (!authReady) return undefined;
    const controller = new AbortController();
    setRecommendations([]);
    setOnboardingRatings({});
    setShowOnboarding(false);
    setSelectedMovie(null);
    setColdStart(false);
    setStrategy("");
    const source = account
      ? { type: "account" }
      : { type: "demo", userId: 1 };
    loadRecommendations(source, controller.signal);
    return () => controller.abort();
  }, [authReady, account?.id, loadRecommendations]);

  useEffect(() => {
    const timer = window.setTimeout(loadMovies, 280);
    return () => window.clearTimeout(timer);
  }, [loadMovies]);

  const submitDemo = (event) => {
    event.preventDefault();
    const parsed = Number(demoUserId);
    if (!/^[1-9]\d*$/.test(demoUserId) || !Number.isSafeInteger(parsed)) {
      setError("Enter a positive MovieLens dataset ID, such as 1 or 101.");
      return;
    }
    setShowOnboarding(false);
    loadRecommendations({ type: "demo", userId: parsed });
    document.getElementById("recommendations")?.scrollIntoView({ behavior: "smooth" });
  };

  const logout = async () => {
    setError("");
    try {
      await api.logout();
      setAccount(null);
      setRecommendations([]);
      setOnboardingRatings({});
      setSelectedMovie(null);
      setShowOnboarding(false);
      setError("You have been signed out.");
    } catch (requestError) {
      setError(requestError.message);
    }
  };

  const handleAuthenticated = (currentAccount) => {
    setAccount(currentAccount);
    setAuthOpen(false);
    setRecommendations([]);
    setSelectedMovie(null);
    setOnboardingRatings({});
    setError("");
  };

  const submitOnboarding = () => {
    const ratings = Object.entries(onboardingRatings).map(([movieId, rating]) => ({ movie_id: Number(movieId), rating }));
    if (ratings.length < 3) return;
    setOnboardingLoading(true);
    setError("");
    api.onboardingRecommendations(ratings, 10)
      .then((payload) => {
        setRecommendations(payload.recommendations);
        setStrategy(payload.strategy);
        setColdStart(false);
        setRecommendationSource("onboarding");
        setShowOnboarding(false);
        setApiOnline(true);
        window.setTimeout(() => document.getElementById("recommendations")?.scrollIntoView({ behavior: "smooth" }), 0);
      })
      .catch((requestError) => { setError(requestError.message); setApiOnline(false); })
      .finally(() => setOnboardingLoading(false));
  };

  const strategyLabel = useMemo(() => {
    if (recommendationSource === "account") return "Your private taste profile";
    if (recommendationSource === "onboarding") return "Temporary onboarding demo";
    return `MovieLens demo · user ${demoUserId}`;
  }, [recommendationSource, demoUserId]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="CineMatch home"><span className="brand-mark"><Icon name="play" size={17} /></span><span>Cine<span>Match</span></span></a>
        <nav aria-label="Main navigation"><a href="#recommendations">For you</a><a href="#discover">Discover</a><a href="#method">How it works</a></nav>
        <div className="topbar-actions">
          <div className={`api-status ${apiOnline === false ? "api-status--offline" : ""}`}><span />{apiOnline === false ? "API offline" : apiOnline ? "Model online" : "Connecting"}</div>
          {account ? (
            <div className="account-menu">
              <span className="account-menu__identity"><Icon name="user" size={16} /><span><strong>{account.display_name}</strong><small>{account.email}</small></span></span>
              <button onClick={logout} disabled={!authReady}>Sign out</button>
            </div>
          ) : READ_ONLY_DEMO ? (
            <span className="topbar-readonly">Read-only demo</span>
          ) : (
            <button className="topbar-auth" onClick={() => setAuthOpen(true)} disabled={!authReady}>{authReady ? "Sign in" : "Checking session…"}</button>
          )}
        </div>
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero-glow hero-glow--one" /><div className="hero-glow hero-glow--two" />
          <div className="hero-copy">
            <span className="eyebrow"><Icon name="sparkles" size={16} /> Collaborative intelligence</span>
            <h1>Less scrolling.<br /><em>More watching.</em></h1>
            <p>{account ? `Welcome, ${account.display_name}. Your private recommendations are generated from ratings saved to your account.` : "CineMatch learns your taste from real rating patterns and surfaces the movies most likely to become your next favorite."}</p>
            {READ_ONLY_DEMO && <div className="demo-mode-banner" role="note">Public read-only demo. Account creation, login, ratings, and watchlist changes are disabled.</div>}
            {!account && (
              <>
                <form className="user-form demo-user-form" onSubmit={submitDemo}>
                  <label htmlFor="demo-user-id">MovieLens demo user ID <span>Public sample data only</span></label>
                  <div><span><Icon name="user" size={19} /></span><input id="demo-user-id" value={demoUserId} onChange={(event) => setDemoUserId(event.target.value)} inputMode="numeric" pattern="[1-9][0-9]*" aria-describedby="demo-user-help" /><button type="submit">Explore demo <Icon name="arrow" size={18} /></button></div>
                  <small id="demo-user-help">This read-only demo ID is separate from your CineMatch account.</small>
                </form>
                <button className="onboarding-trigger" onClick={() => { setShowOnboarding(true); window.setTimeout(() => document.getElementById("onboarding")?.scrollIntoView({ behavior: "smooth" }), 0); }}><Icon name="sparkles" size={16} /> New here? Try a temporary taste profile</button>
              </>
            )}
            {account && <button className="onboarding-trigger" onClick={() => document.getElementById("discover")?.scrollIntoView({ behavior: "smooth" })}><Icon name="sparkles" size={16} /> Explore movies and add ratings to your account</button>}
            <div className="hero-proof"><div><strong>100K+</strong><span>real ratings</span></div><div><strong>9,742</strong><span>movies explored</span></div><div><strong>0.375</strong><span>Precision@10</span></div></div>
          </div>
          <div className="hero-stack" aria-hidden="true">
            {recommendations.slice(0, 3).map((movie, index) => <div className={`hero-poster hero-poster--${index + 1}`} key={movie.movie_id}><MovieArtwork movie={movie} /><span>{movie.title}</span></div>)}
            <div className="match-orbit"><span>AI match</span><strong>{coldStart ? "Fresh" : "98%"}</strong></div>
          </div>
        </section>

        {error && <div className="error-banner" role="alert"><strong>We hit a snag.</strong><span>{error}</span><button onClick={() => setError("")} aria-label="Dismiss error"><Icon name="close" size={18} /></button></div>}

        {showOnboarding && !account && <TasteOnboarding movies={onboardingMovies} ratings={onboardingRatings} onRate={(movieId, rating) => setOnboardingRatings((current) => ({ ...current, [movieId]: rating }))} onSubmit={submitOnboarding} loading={onboardingLoading} onClose={() => setShowOnboarding(false)} />}

        <section className="content-section" id="recommendations">
          <div className="section-heading"><div><span className="eyebrow">{strategyLabel}</span><h2>{coldStart ? "A great place to begin" : recommendationSource === "account" ? "Picked for your taste" : "Discover a new favorite"}</h2><p>{recommendationSource === "account" ? "These recommendations use only ratings saved to your signed-in account." : recommendationSource === "onboarding" ? `A temporary demo based on ${Object.keys(onboardingRatings).length} ratings. They have not been saved to an account.` : "These read-only recommendations use public MovieLens sample data and never access account records."}</p></div><span className="model-pill"><Icon name="sparkles" size={15} /> {strategy || (authReady ? "Loading model" : "Restoring session")}</span></div>
          {loadingRecommendations || !authReady ? <SkeletonGrid count={5} /> : recommendations.length ? <div className="movie-grid">{recommendations.map((movie, index) => <MovieCard movie={movie} rank={index + 1} onSelect={setSelectedMovie} key={movie.movie_id} />)}</div> : <EmptyState title="No recommendations yet" message={recommendationSource === "account" ? "Rate a few movies to start shaping your recommendations." : "Choose another MovieLens demo ID to see a sample list."} />}
        </section>

        <section className="content-section discover" id="discover">
          <div className="section-heading"><div><span className="eyebrow">Explore the catalog</span><h2>Find something specific</h2><p>Search all 9,742 titles or narrow the collection by genre.</p></div></div>
          <div className="discovery-tools"><label className="search-box"><Icon name="search" size={19} /><span className="sr-only">Search movies</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search by movie title…" /></label><div className="genre-scroller">{GENRES.map((item) => <button className={genre === item ? "active" : ""} onClick={() => setGenre(item)} key={item}>{item}</button>)}</div></div>
          {loadingMovies ? <SkeletonGrid count={5} /> : movies.length ? <div className="movie-grid movie-grid--discover">{movies.map((movie) => <MovieCard movie={movie} onSelect={setSelectedMovie} key={movie.movie_id} />)}</div> : <EmptyState title="No movies found" message="Try a shorter title or choose another genre." />}
        </section>

        <section className="method-section" id="method">
          <div><span className="eyebrow">Inside the match</span><h2>Built on taste,<br />not just trends.</h2><p>The selected 20-factor SVD model maps viewers and movies into the same latent taste space. That lets CineMatch predict what you’ll enjoy—even when two films don’t look similar on the surface.</p></div>
          <div className="method-steps"><article><span>01</span><h3>Learn your signal</h3><p>Your existing ratings establish a unique preference profile.</p></article><article><span>02</span><h3>Find hidden patterns</h3><p>Collaborative filtering discovers shared taste across 610 viewers.</p></article><article><span>03</span><h3>Rank the unseen</h3><p>Already-rated movies are removed before your list is scored.</p></article></div>
        </section>
      </main>

      <footer><a className="brand" href="#top"><span className="brand-mark"><Icon name="play" size={15} /></span><span>Cine<span>Match</span></span></a><p>Movie discovery, powered by collaborative filtering.</p><span>MovieLens Small · Internship project</span></footer>
      {selectedMovie && <MovieModal key={`${selectedMovie.movie_id}-${account?.id ?? "guest"}`} movie={selectedMovie} account={account} demoUserId={demoUserId} readOnlyDemo={READ_ONLY_DEMO} onRatingSaved={() => loadRecommendations({ type: "account" })} onClose={() => setSelectedMovie(null)} onLoginRequired={() => { setSelectedMovie(null); setAuthOpen(true); }} />}
      {authOpen && !READ_ONLY_DEMO && <AuthDialog onClose={() => setAuthOpen(false)} onAuthenticated={handleAuthenticated} />}
    </div>
  );
}
