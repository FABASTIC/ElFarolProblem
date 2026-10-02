import { Component, type ErrorInfo, type ReactNode } from "react";

interface BoundaryProps {
  fallback: (error: Error, retry: () => void) => ReactNode;
  resetKey?: string;
  children: ReactNode;
}

interface BoundaryState {
  error: Error | null;
}

export default class Boundary extends Component<BoundaryProps, BoundaryState> {
  state: BoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): BoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[control-center]", error, info.componentStack);
  }

  componentDidUpdate(previous: BoundaryProps) {
    if (previous.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null });
  }

  retry = () => this.setState({ error: null });

  render() {
    return this.state.error ? this.props.fallback(this.state.error, this.retry) : this.props.children;
  }
}
