import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import Boundary from "./components/Boundary";
import "./styles.css";

const container = document.getElementById("root");

if (container) {
  createRoot(container).render(
    <StrictMode>
      <Boundary
        fallback={(error, retry) => (
          <main className="fault" role="alert">
            <p className="eyebrow">EL FAROL // CONTROL CENTER FAULT</p>
            <p className="fault__message">{error.message}</p>
            <button type="button" className="btn" onClick={retry}>
              RESTART INTERFACE
            </button>
          </main>
        )}
      >
        <App />
      </Boundary>
    </StrictMode>,
  );
}
