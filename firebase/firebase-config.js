import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getAnalytics }  from "https://www.gstatic.com/firebasejs/10.12.2/firebase-analytics.js";
import { getAuth }       from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";

async function initFirebase() {
  const response = await fetch("/firebase-config");
  if (!response.ok) {
    throw new Error(`Failed to load Firebase config: ${response.status} ${response.statusText}`);
  }
  const config = await response.json();
  if (!config.configured) {
    console.warn("Firebase not configured. Auth features will be disabled.", config.missing);
    return { app: null, auth: null, analytics: null, configured: false };
  }
  const { configured, ...firebaseConfig } = config;
  const app       = initializeApp(firebaseConfig);
  const analytics = getAnalytics(app);
  const auth      = getAuth(app);
  return { app, auth, analytics, configured: true };
}

export const firebaseReady = initFirebase();
