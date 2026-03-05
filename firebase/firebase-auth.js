
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut as firebaseSignOut,
  onAuthStateChanged,
  sendPasswordResetEmail,
  updateProfile,
  GoogleAuthProvider,
  signInWithPopup,
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";

import { firebaseReady } from "./firebase-config.js";

// Resolve auth instance once Firebase is initialised
const getAuth = async () => (await firebaseReady).auth;

// ─── Email / Password ────────────────────────────────────────────────────────

/**
 * Register a new user with email & password.
 * @param {string} email
 * @param {string} password
 * @param {string} [displayName] - Optional display name set on the profile.
 * @returns {Promise<import("firebase/auth").UserCredential>}
 */
export async function signUp(email, password, displayName = "") {
  const auth = await getAuth();
  const credential = await createUserWithEmailAndPassword(auth, email, password);
  if (displayName) {
    await updateProfile(credential.user, { displayName });
  }
  return credential;
}

/**
 * Sign in an existing user with email & password.
 * @param {string} email
 * @param {string} password
 * @returns {Promise<import("firebase/auth").UserCredential>}
 */
export async function signIn(email, password) {
  const auth = await getAuth();
  return signInWithEmailAndPassword(auth, email, password);
}

/**
 * Sign out the currently authenticated user.
 * @returns {Promise<void>}
 */
export async function signOut() {
  const auth = await getAuth();
  return firebaseSignOut(auth);
}

// ─── Google OAuth ─────────────────────────────────────────────────────────────

const googleProvider = new GoogleAuthProvider();

/**
 * Sign in via Google popup.
 * @returns {Promise<import("firebase/auth").UserCredential>}
 */
export async function signInWithGoogle() {
  const auth = await getAuth();
  return signInWithPopup(auth, googleProvider);
}

// ─── Password Reset ───────────────────────────────────────────────────────────

/**
 * Send a password-reset e-mail.
 * @param {string} email
 * @returns {Promise<void>}
 */
export async function resetPassword(email) {
  const auth = await getAuth();
  return sendPasswordResetEmail(auth, email);
}

// ─── Auth State ───────────────────────────────────────────────────────────────

/**
 * Subscribe to authentication state changes.
 * @param {(user: import("firebase/auth").User | null) => void} callback
 * @returns {Promise<import("firebase/auth").Unsubscribe>}
 */
export async function onUserStateChanged(callback) {
  const auth = await getAuth();
  return onAuthStateChanged(auth, callback);
}

/**
 * Return the currently signed-in user, or null if not authenticated.
 * @returns {Promise<import("firebase/auth").User | null>}
 */
export async function getCurrentUser() {
  const auth = await getAuth();
  return auth.currentUser;
}
