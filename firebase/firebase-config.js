// firebase/firebase-config.js
// Firebase config is fetched from the FastAPI backend (/firebase-config),
// which reads values from the server-side .env file.
// Nothing is hard-coded here — secrets stay on the server.

import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getAnalytics }  from "https://www.gstatic.com/firebasejs/10.12.2/firebase-analytics.js";
import { getAuth }       from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";

/**
 * Fetch Firebase client config from the FastAPI backend.
 * The backend reads values from .env so they are never exposed in source code.
 * @returns {Promise<{app, auth, analytics}>}
 */
async function initFirebase() {
  const response = await fetch("/firebase-config");
  if (!response.ok) {
    throw new Error(`Failed to load Firebase config: ${response.status} ${response.statusText}`);
  }
  const firebaseConfig = await response.json();

  const app       = initializeApp(firebaseConfig);
  const analytics = getAnalytics(app);
  const auth      = getAuth(app);

  return { app, auth, analytics };
}

// Singleton promise — resolves once Firebase is ready.
// Import and await this in any module that needs auth/analytics.
//
// Example:
//   import { firebaseReady } from "./firebase-config.js";
//   const { auth } = await firebaseReady;
//
export const firebaseReady = initFirebase();
