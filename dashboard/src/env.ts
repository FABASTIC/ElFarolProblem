export const HOSTED = import.meta.env.MODE === "hosted" || import.meta.env.VITE_ELFAROL_HOSTED === "1";

export const BROWSER_ENGINE =
  HOSTED || import.meta.env.VITE_ELFAROL_ENGINE === "browser" || (typeof window !== "undefined" && new URLSearchParams(window.location.search).get("engine") === "browser");

export const REPO_URL = "https://github.com/FABASTIC/ElFarolProblem";
